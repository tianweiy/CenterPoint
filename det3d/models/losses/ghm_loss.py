#####################
# THIS LOSS IS NOT WORKING!!!!
#####################
"""
The implementation of GHM-C and GHM-R losses.
Details can be found in the paper `Gradient Harmonized Single-stage Detector`:
https://arxiv.org/abs/1811.05181
Copyright (c) 2018 Multimedia Laboratory, CUHK.
Licensed under the MIT License (see LICENSE for details)
Written by Buyu Li
"""

import torch
from det3d.models.losses.losses import Loss, _sigmoid_cross_entropy_with_logits


class GHMCLoss(Loss):
    def __init__(self, bins=10, momentum=0):
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.count = 50

    def _compute_loss(
        self, prediction_tensor, target_tensor, weights, class_indices=None
    ):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        input = prediction_tensor
        target = target_tensor
        batch_size = prediction_tensor.shape[0]
        num_anchors = prediction_tensor.shape[1]
        num_class = prediction_tensor.shape[2]

        edges = self.edges
        weights_ghm = torch.zeros_like(input).view(-1, num_class)
        per_entry_cross_ent = _sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor
        )

        # gradient length
        g = torch.abs(input.sigmoid().detach() - target).view(-1, num_class)

        valid = weights.view(-1, 1).expand(-1, num_class) >= 0
        num_examples = max(valid.float().sum().item(), 1.0)
        num_valid_bins = 0  # n valid bins

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = (
                        self.momentum * self.acc_sum[i]
                        + (1 - self.momentum) * num_in_bin
                    )
                    weights_ghm[inds] = num_examples / self.acc_sum[i]
                else:
                    weights_ghm[inds] = num_examples / num_in_bin
                num_valid_bins += 1

        if num_valid_bins > 0:
            weights_ghm = weights_ghm / num_valid_bins

        loss = per_entry_cross_ent * weights_ghm.view(
            batch_size, num_anchors, num_class
        )

        # loss = torch.nn.BCEWithLogitsLoss(
        #     weight=weights_ghm.view(batch_size, num_anchors, num_class),
        #     reduction='none',
        # )(prediction_tensor, target_tensor)

        return loss


class GHMRLoss(Loss):
    def __init__(self, mu=0.02, bins=10, momentum=0, code_weights=None):
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self._codewise = True

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        # ASL1 loss
        diff = prediction_tensor - target_tensor
        loss = torch.sqrt(diff * diff + self.mu * self.mu) - self.mu
        batch_size = prediction_tensor.shape[0]
        num_anchors = prediction_tensor.shape[1]
        num_codes = prediction_tensor.shape[2]

        # gradient length
        g = (
            torch.abs(diff / torch.sqrt(self.mu * self.mu + diff * diff))
            .detach()
            .view(-1, num_codes)
        )
        weights_ghm = torch.zeros_like(g)

        valid = weights.view(-1, 1).expand(-1, num_codes) > 0
        # print(g.shape, prediction_tensor.shape, valid.shape)
        num_examples = max(valid.float().sum().item() / num_codes, 1.0)
        num_valid_bins = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                num_valid_bins += 1
                if self.momentum > 0:
                    self.acc_sum[i] = (
                        self.momentum * self.acc_sum[i]
                        + (1 - self.momentum) * num_in_bin
                    )
                    weights_ghm[inds] = num_examples / self.acc_sum[i]
                else:
                    weights_ghm[inds] = num_examples / num_in_bin
        if num_valid_bins > 0:
            weights_ghm /= num_valid_bins
        weights_ghm = weights_ghm.view(batch_size, num_anchors, num_codes)
        loss = loss * weights_ghm / num_examples
        return loss
