"""Classification and regression loss functions for object detection.

Localization losses:
 * WeightedL2LocalizationLoss
 * WeightedSmoothL1LocalizationLoss

Classification losses:
 * WeightedSigmoidClassificationLoss
 * WeightedSoftmaxClassificationLoss
 * BootstrappedSigmoidClassificationLoss
"""
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..registry import LOSSES
from .utils import weight_reduce_loss


def indices_to_dense_vector(
    indices, size, indices_value=1.0, default_value=0, dtype=np.float32
):
    """Creates dense vector with indices set to specific value and rest to zeros.

    This function exists because it is unclear if it is safe to use
        tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

    Args:
        indices: 1d Tensor with integer indices which are to be set to
            indices_values.
        size: scalar with size (integer) of output Tensor.
        indices_value: values of elements specified by indices in the output vector
        default_value: values of other elements in the output vector.
        dtype: data type.

    Returns:
        dense 1D Tensor of shape [size] with indices set to indices_values and the
            rest set to default_value.
    """
    dense = torch.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense


class Loss(object):
    """Abstract base class for loss functions."""

    __metaclass__ = ABCMeta

    def __call__(
        self,
        prediction_tensor,
        target_tensor,
        ignore_nan_targets=False,
        scope=None,
        **params
    ):
        """Call the loss function.

        Args:
        prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
            representing predicted quantities.
        target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
            regression or classification targets.
        ignore_nan_targets: whether to ignore nan targets in the loss computation.
            E.g. can be used if the target tensor is missing groundtruth data that
            shouldn't be factored into the loss.
        scope: Op scope name. Defaults to 'Loss' if None.
        **params: Additional keyword arguments for specific implementations of
                the Loss.

        Returns:
        loss: a tensor representing the value of the loss function.
        """
        if ignore_nan_targets:
            target_tensor = torch.where(
                torch.isnan(target_tensor), prediction_tensor, target_tensor
            )
        return self._compute_loss(prediction_tensor, target_tensor, **params)

    @abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """Method to be overridden by implementations.

        Args:
        prediction_tensor: a tensor representing predicted quantities
        target_tensor: a tensor representing regression or classification targets
        **params: Additional keyword arguments for specific implementations of
                the Loss.

        Returns:
        loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
            anchor
        """
        pass


@LOSSES.register_module
class WeightedL2LocalizationLoss(Loss):
    """L2 localization loss function with anchorwise output support.

    Loss[b,a] = .5 * ||weights[b,a] * (prediction[b,a,:] - target[b,a,:])||^2
    """

    def __init__(self, code_weights=None, device=None):
        super().__init__()
        if code_weights is not None:
            self._code_weights = np.array(code_weights, dtype=np.float32)
            self._code_weights = Variable(
                torch.from_numpy(self._code_weights).to(device)
            )
        else:
            self._code_weights = None

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets
        weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:
            self._code_weights = self._code_weights.type_as(prediction_tensor)
            self._code_weights = self._code_weights.view(1, 1, -1)
            diff = self._code_weights * diff
        weighted_diff = diff * weights.unsqueeze(-1)
        square_diff = 0.5 * weighted_diff * weighted_diff
        return square_diff.sum(2)


@LOSSES.register_module
class WeightedSmoothL1Loss(nn.Module):
    """Smooth L1 localization loss function.

    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(
        self,
        sigma=3.0,
        reduction="mean",
        code_weights=None,
        codewise=True,
        loss_weight=1.0,
        raw_l1=False
    ):
        super(WeightedSmoothL1Loss, self).__init__()
        
        # just l1 loss
        self.raw_l1 = raw_l1 

        print("Raw L1 Loss:", raw_l1)
        
        self._sigma = sigma

        if code_weights is not None:
            self._code_weights = torch.tensor(code_weights,
                                              dtype=torch.float32)
        else:
            self._code_weights = None

        # self._code_weights = None

        self._codewise = codewise
        self._reduction = reduction
        self._loss_weight = loss_weight

    def forward(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets
        weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:
            # code_weights = self._code_weights.type_as(prediction_tensor).to(diff.device)
            diff = self._code_weights.view(1, 1, -1).to(diff.device) * diff
        abs_diff = torch.abs(diff)

        if not self.raw_l1:
            abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma ** 2)).type_as(abs_diff)
            loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) + (
                abs_diff - 0.5 / (self._sigma ** 2)
            ) * (1.0 - abs_diff_lt_1)
        else:
            loss = abs_diff

        if self._codewise:
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
        else:
            anchorwise_smooth_l1norm = torch.sum(loss, 2)  #  * weights
            if weights is not None:
                anchorwise_smooth_l1norm *= weights

        return anchorwise_smooth_l1norm


@LOSSES.register_module
class WeightedL1Loss(nn.Module):
    """L1 localization loss function.
    """

    def __init__(
        self,
        reduction="mean",
        code_weights=None,
        codewise=True,
        loss_weight=1.0,
    ):
        super(WeightedL1Loss, self).__init__()

        if code_weights is not None:
            self._code_weights = torch.tensor(code_weights,
                                              dtype=torch.float32)
        else:
            self._code_weights = None

        self._codewise = codewise
        self._reduction = reduction
        self._loss_weight = loss_weight

    def forward(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets
        weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:
            diff = self._code_weights.view(1, 1, -1).to(diff.device) * diff
        loss = torch.abs(diff)

        if self._codewise:
            anchorwise_l1norm = loss
            if weights is not None:
                anchorwise_l1norm *= weights.unsqueeze(-1)
        else:
            anchorwise_l1norm = torch.sum(loss, 2)  #  * weights
            if weights is not None:
                anchorwise_l1norm *= weights

        return anchorwise_l1norm

def _sigmoid_cross_entropy_with_logits(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    # transpose_param = [0] + [param[-1]] + param[1:-1]
    # logits = logits.permute(*transpose_param)
    # loss_ftor = nn.NLLLoss(reduce=False)
    # loss = loss_ftor(F.logsigmoid(logits), labels)
    return loss


def _softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)  # [N, ..., C] -> [N, C, ...]
    loss_ftor = nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


@LOSSES.register_module
class WeightedSigmoidClassificationLoss(Loss):
    """Sigmoid cross entropy classification loss function."""

    def _compute_loss(
        self, prediction_tensor, target_tensor, weights, class_indices=None
    ):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]
        class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        weights = weights.unsqueeze(-1)
        if class_indices is not None:
            weights *= (
                indices_to_dense_vector(class_indices, prediction_tensor.shape[2])
                .view(1, 1, -1)
                .type_as(prediction_tensor)
            )
        per_entry_cross_ent = _sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor
        )
        return per_entry_cross_ent * weights


@LOSSES.register_module
class SigmoidFocalLoss(nn.Module):
    """Sigmoid focal cross entropy loss.

    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0):
        """Constructor.

        Args:
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.
        all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super(SigmoidFocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction
        self._loss_weight = loss_weight

    def forward(
        self, prediction_tensor, target_tensor, weights=None, class_indices=None
    ):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]
        class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        weights = weights.unsqueeze(2)
        if class_indices is not None:
            weights *= (
                indices_to_dense_vector(class_indices, prediction_tensor.shape[2])
                .view(1, 1, -1)
                .type_as(prediction_tensor)
            )
        per_entry_cross_ent = _sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor
        )
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = (target_tensor * prediction_probabilities) + (
            (1 - target_tensor) * (1 - prediction_probabilities)
        )
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = target_tensor * self._alpha + (1 - target_tensor) * (
                1 - self._alpha
            )

        focal_cross_entropy_loss = (
            modulating_factor * alpha_weight_factor * per_entry_cross_ent
        )
        return focal_cross_entropy_loss * weights


@LOSSES.register_module
class SoftmaxFocalClassificationLoss(Loss):
    """Softmax focal cross entropy loss.

    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.

        Args:
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.
        """
        self._alpha = alpha
        self._gamma = gamma

    def _compute_loss(
        self, prediction_tensor, target_tensor, weights, class_indices=None
    ):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]
        class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        weights = weights.unsqueeze(2)
        if class_indices is not None:
            weights *= (
                indices_to_dense_vector(class_indices, prediction_tensor.shape[2])
                .view(1, 1, -1)
                .type_as(prediction_tensor)
            )
        per_entry_cross_ent = _softmax_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor
        )
        # convert [N, num_anchors] to [N, num_anchors, num_classes]
        per_entry_cross_ent = per_entry_cross_ent.unsqueeze(-1) * target_tensor
        prediction_probabilities = F.softmax(prediction_tensor, dim=-1)
        p_t = (target_tensor * prediction_probabilities) + (
            (1 - target_tensor) * (1 - prediction_probabilities)
        )
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = torch.where(
                target_tensor[..., 0] == 1,
                torch.tensor(1 - self._alpha).type_as(per_entry_cross_ent),
                torch.tensor(self._alpha).type_as(per_entry_cross_ent),
            )
        focal_cross_entropy_loss = (
            modulating_factor * alpha_weight_factor * per_entry_cross_ent
        )
        return focal_cross_entropy_loss * weights


@LOSSES.register_module
class WeightedSoftmaxClassificationLoss(nn.Module):
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0, loss_weight=1.0, name=""):
        """Constructor.

        Args:
        logit_scale: When this value is high, the prediction is "diffused" and
                    when this value is low, the prediction is made peakier.
                    (default 1.0)

        """
        super(WeightedSoftmaxClassificationLoss, self).__init__()
        self.name = name
        self._loss_weight = loss_weight
        self._logit_scale = logit_scale

    def forward(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors]
            representing the value of the loss function.
        """
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = torch.div(prediction_tensor, self._logit_scale)
        per_row_cross_ent = _softmax_cross_entropy_with_logits(
            labels=target_tensor.view(-1, num_classes),
            logits=prediction_tensor.view(-1, num_classes),
        )

        return per_row_cross_ent.view(weights.shape) * weights


@LOSSES.register_module
class BootstrappedSigmoidClassificationLoss(Loss):
    """Bootstrapped sigmoid cross entropy classification loss function.

    This loss uses a convex combination of training labels and the current model's
    predictions as training targets in the classification loss. The idea is that
    as the model improves over time, its predictions can be trusted more and we
    can use these predictions to mitigate the damage of noisy/incorrect labels,
    because incorrect labels are likely to be eventually highly inconsistent with
    other stimuli predicted to have the same label by the model.

    In "soft" bootstrapping, we use all predicted class probabilities, whereas in
    "hard" bootstrapping, we use the single class favored by the model.

    See also Training Deep Neural Networks On Noisy Labels with Bootstrapping by
    Reed et al. (ICLR 2015).
    """

    def __init__(self, alpha, bootstrap_type="soft"):
        """Constructor.

        Args:
        alpha: a float32 scalar tensor between 0 and 1 representing interpolation
            weight
        bootstrap_type: set to either 'hard' or 'soft' (default)

        Raises:
        ValueError: if bootstrap_type is not either 'hard' or 'soft'
        """
        if bootstrap_type != "hard" and bootstrap_type != "soft":
            raise ValueError(
                "Unrecognized bootstrap_type: must be one of " "'hard' or 'soft.'"
            )
        self._alpha = alpha
        self._bootstrap_type = bootstrap_type

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        if self._bootstrap_type == "soft":
            bootstrap_target_tensor = self._alpha * target_tensor + (
                1.0 - self._alpha
            ) * torch.sigmoid(prediction_tensor)
        else:
            bootstrap_target_tensor = (
                self._alpha * target_tensor
                + (1.0 - self._alpha) * (torch.sigmoid(prediction_tensor) > 0.5).float()
            )
        per_entry_cross_ent = _sigmoid_cross_entropy_with_logits(
            labels=bootstrap_target_tensor, logits=prediction_tensor
        )
        return per_entry_cross_ent * weights.unsqueeze(2)
