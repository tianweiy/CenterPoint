# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()

  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()
  isnotnan = (~ torch.isnan(gt_regr)).float()
  mask *= isnotnan
  regr = regr * mask
  gt_regr = gt_regr * mask

  loss = torch.abs(regr - gt_regr)
  loss = loss.transpose(2, 0)

  loss = torch.sum(loss, dim=2)
  loss = torch.sum(loss, dim=1)
  # else:
  #  # D x M x B 
  #  loss = loss.reshape(loss.shape[0], -1)

  loss = loss / (num + 1e-4)
  # import pdb; pdb.set_trace()
  return loss 


def _smooth_reg_loss(regr, gt_regr, mask, sigma=3):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()
  isnotnan = (~ torch.isnan(gt_regr)).float()
  mask *= isnotnan
  regr = regr * mask
  gt_regr = gt_regr * mask
  
  abs_diff = torch.abs(regr - gt_regr)

  abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
  
  loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + (
      abs_diff - 0.5 / (sigma ** 2)
  ) * (1.0 - abs_diff_lt_1)

  loss = loss.transpose(2, 0)

  loss = torch.sum(loss, dim=2)
  loss = torch.sum(loss, dim=1)
  # else:
  #  # D x M x B 
  #  loss = loss.reshape(loss.shape[0], -1)

  loss = loss / (num + 1e-4)
  # import pdb; pdb.set_trace()
  return loss 


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class SmoothRegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(SmoothRegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target, sin_loss):
    assert sin_loss is False
    pred = _transpose_and_gather_feat(output, ind)
    loss = _smooth_reg_loss(pred, target, mask)
    return loss




def _reg_cls_loss(regr, gt_regr, mask, is_reduce=True):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  regr = regr.squeeze(-1)
  gt_regr = gt_regr.float()
  num = mask.float().sum()

  regr = regr[mask, :]
  gt_regr = gt_regr[mask]

  if len(gt_regr) > 0:
    loss = torch.nn.functional.cross_entropy(regr.reshape(-1, regr.shape[-1]), gt_regr.long().reshape(-1), reduction='sum')
  else:
    loss = (0 * mask).sum() 

  loss = loss / (num + 1e-4)
  # import pdb; pdb.set_trace()
  return loss 



class RegClsLoss(nn.Module):
  '''Regression CE loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegClsLoss, self).__init__()
  
  def forward(self, output, mask, ind, target, is_reduce=True):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_cls_loss(pred, target, mask, is_reduce)
    return loss


class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self, opt=None):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat, is_reduce=True):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)

    if is_reduce:
      pos_loss = pos_loss.sum()
      if num_pos == 0:
        return - neg_loss
      return - (pos_loss + neg_loss) / num_pos
    else:
      if num_pos == 0:
        return -pos_loss * 0, - neg_loss.sum()
      else:
        return -pos_loss/num_pos, -neg_loss / num_pos
