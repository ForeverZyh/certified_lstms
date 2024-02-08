"""Interval bound propagation layers in pytorch."""
import sys
import numpy as np
import queue
import torch
from functools import partial
from itertools import product
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from perturbation import Perturbation

# DEBUG = False
DEBUG = True
TOLERANCE = 1e-5


##### BoundedTensor and subclasses #####

class BoundedTensor(object):
    """Contains a torch.Tensor plus bounds on it."""

    @property
    def shape(self):
        return self.val.shape

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)


class IntervalBoundedTensor(BoundedTensor):
    """A tensor with elementwise upper and lower bounds.
    This is the main building BoundedTensor subclass.
    All layers in this library accept IntervalBoundedTensor as input,
    and when handed one will generate another IntervalBoundedTensor as output.
    """

    def __init__(self, val, lb, ub):
        self.val = val
        self.lb = lb
        self.ub = ub
        if DEBUG:
            # Sanity check lower and upper bounds when creating this
            # Note that there may be small violations on the order of 1e-5,
            # due to floating point rounding/non-associativity issues.
            # e.g. https://github.com/pytorch/pytorch/issues/9146
            max_lb_violation = torch.max((lb - val) * (lb <= ub).float())  # exclude bottom
            if max_lb_violation > TOLERANCE:
                print('WARNING: Lower bound wrong (max error = %g)' % max_lb_violation.item(), file=sys.stderr)
            max_ub_violation = torch.max((val - ub) * (lb <= ub).float())  # exclude bottom
            if max_ub_violation > TOLERANCE:
                print('WARNING: Upper bound wrong (max error = %g)' % max_ub_violation.item(), file=sys.stderr)

    @staticmethod
    def bottom_like(x):
        return IntervalBoundedTensor(torch.zeros_like(x), torch.ones_like(x) * 1e16, torch.ones_like(x) * (-1e16))

    @staticmethod
    def bottom(x, device):
        return IntervalBoundedTensor(torch.zeros(x, device=device), torch.ones(x, device=device) * 1e16,
                                     torch.ones(x, device=device) * (-1e16))

    @staticmethod
    def point(x):
        return IntervalBoundedTensor(x, x.clone(), x.clone())

    def float(self):
        return IntervalBoundedTensor(self.val.float(), self.lb.float(), self.ub.float())

    ### Reimplementations of torch.Tensor methods
    def __neg__(self):
        return IntervalBoundedTensor(-self.val, -self.ub, -self.lb)

    def permute(self, *dims):
        return IntervalBoundedTensor(self.val.permute(*dims),
                                     self.lb.permute(*dims),
                                     self.ub.permute(*dims))

    def squeeze(self, dim=None):
        return IntervalBoundedTensor(self.val.squeeze(dim=dim),
                                     self.lb.squeeze(dim=dim),
                                     self.ub.squeeze(dim=dim))

    def unsqueeze(self, dim):
        return IntervalBoundedTensor(self.val.unsqueeze(dim),
                                     self.lb.unsqueeze(dim),
                                     self.ub.unsqueeze(dim))

    def flip(self, dim):
        return IntervalBoundedTensor(self.val.flip(dim),
                                     self.lb.flip(dim),
                                     self.ub.flip(dim))

    def merge(self, x2):
        if isinstance(x2, IntervalBoundedTensor):
            val = torch.where(self.lb <= self.ub, self.val, x2.val)
            return IntervalBoundedTensor(val, torch.min(self.lb, x2.lb), torch.max(self.ub, x2.ub))
        else:
            val = torch.where(self.lb <= self.ub, self.val, x2)
            return IntervalBoundedTensor(val, torch.min(self.lb, x2), torch.max(self.ub, x2))

    def clone(self):
        return IntervalBoundedTensor(self.val.clone(), self.lb.clone(), self.ub.clone())

    def repeat(self, *dim):
        return IntervalBoundedTensor(self.val.repeat(*dim), self.lb.repeat(*dim), self.ub.repeat(*dim))

    def view(self, *dim):
        return IntervalBoundedTensor(self.val.view(*dim), self.lb.view(*dim), self.ub.view(*dim))

    def reshape(self, *dim):
        return IntervalBoundedTensor(self.val.reshape(*dim), self.lb.reshape(*dim), self.ub.reshape(*dim))

    def to(self, device):
        self.val = self.val.to(device)
        self.lb = self.lb.to(device)
        self.ub = self.ub.to(device)
        return self

    # For slicing
    def __getitem__(self, key):
        return IntervalBoundedTensor(self.val.__getitem__(key),
                                     self.lb.__getitem__(key),
                                     self.ub.__getitem__(key))

    def __setitem__(self, key, value):
        if not isinstance(value, IntervalBoundedTensor):
            raise TypeError(value)
        self.val.__setitem__(key, value.val)
        self.lb.__setitem__(key, value.lb)
        self.ub.__setitem__(key, value.ub)

    def __delitem__(self, key):
        self.val.__delitem__(key)
        self.lb.__delitem__(key)
        self.ub.__delitem__(key)


class DiscreteChoiceTensorWithUNK(BoundedTensor):
    """A tensor for which each row can take a discrete set of values.

    More specifically, each slice along the first d-1 dimensions of the tensor
    is allowed to take on values within some discrete set.
    The overall tensor's possible values are the direct product of all these
    individual choices.

    Recommended usage is only as an input tensor, passed to Linear() layer.
    Only some layers accept this tensor.
    """

    def __init__(self, val, choice_mat, choice_mask, sequence_mask, unk_mask):
        """Create a DiscreteChoiceTensor.

        Args:
          val: value, dimension (*, d).  Let m = product of first d-1 dimensions.
          choice_mat: all choices-padded with 0 where fewer than max choices are available, size (*, C, d)
          choice_mask: mask tensor s.t. choice_maks[i,j,k]==1 iff choice_mat[i,j,k,:] is a valid choice, size (*, C)
          sequence_mask: mask tensor s.t. sequence_mask[i,j,k]==1 iff choice_mat[i,j] is a valid word in a sequence and not padding, size (*)
        """
        self.val = val
        self.choice_mat = choice_mat
        self.choice_mask = choice_mask
        self.sequence_mask = sequence_mask
        self.unk_mask = unk_mask

    def to_interval_bounded(self, eps=1.0):
        """
        Convert to an IntervalBoundedTensor.
        Args:
          - eps: float, scaling factor for the interval bounds
        """
        choice_mask_mat = (((1 - self.choice_mask).float() * 1e16)).unsqueeze(-1)  # *, C, 1
        seq_mask_mat = self.sequence_mask.unsqueeze(-1).unsqueeze(-1).float()
        lb = torch.min((self.choice_mat + choice_mask_mat) * seq_mask_mat, -2)[0] * self.sequence_mask.unsqueeze(-1)
        ub = torch.max((self.choice_mat - choice_mask_mat) * seq_mask_mat, -2)[0] * self.sequence_mask.unsqueeze(-1)
        val = self.val * self.sequence_mask.unsqueeze(-1)
        if eps != 1.0:
            lb = val - (val - lb) * eps
            ub = val + (ub - val) * eps
        return IntervalBoundedTensor(val, lb, ub)

    def to(self, device):
        """Moves the Tensor to the given device"""
        self.val = self.val.to(device)
        self.choice_mat = self.choice_mat.to(device)
        self.choice_mask = self.choice_mask.to(device)
        self.sequence_mask = self.sequence_mask.to(device)
        self.unk_mask = self.unk_mask.to(device)
        return self


class DiscreteChoiceTensorTrans(BoundedTensor):
    """A tensor for which each row can take a discrete set of values.

    More specifically, each slice along the first d-1 dimensions of the tensor
    is allowed to take on values within some discrete set.
    The overall tensor's possible values are the direct product of all these
    individual choices.

    Recommended usage is only as an input tensor, passed to Linear() layer.
    Only some layers accept this tensor.
    """

    def __init__(self, choice_mat, choice_mask, sequence_mask):
        """Create a DiscreteChoiceTensor.

        Args:
          choice_mat: all choices-padded with 0 where fewer than max choices are available, size (*, tran.t, C, d)
          choice_mask: mask tensor s.t. choice_maks[i,j,k]==1 iff choice_mat[i,j,k,:] is a valid choice, size (*, tran.t, C)
          sequence_mask: mask tensor s.t. sequence_mask[i,j,k]==1 iff choice_mat[i,j] is a valid word in a sequence and not padding, size (*)
        """
        self.choice_mat = choice_mat
        self.choice_mask = choice_mask
        self.sequence_mask = sequence_mask

    def to_interval_bounded(self, eps=1.0):
        """
        Convert to an IntervalBoundedTensor.
        Args:
          - eps: float, scaling factor for the interval bounds
        """
        choice_mask_mat = (((1 - self.choice_mask).float() * 1e16)).unsqueeze(-1)  # *, tran.t, C, 1
        seq_mask_mat = self.sequence_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        # lb, ub are in the shape of (*, tran.t, d)
        lb = torch.min((self.choice_mat + choice_mask_mat) * seq_mask_mat, -2)[0] * seq_mask_mat.squeeze(-1)
        ub = torch.max((self.choice_mat - choice_mask_mat) * seq_mask_mat, -2)[0] * seq_mask_mat.squeeze(-1)
        val = (lb + ub) / 2  # use the middle point as the val
        if eps != 1.0:
            lb = val - (val - lb) * eps
            ub = val + (ub - val) * eps
        return IntervalBoundedTensor(val, lb, ub)

    def to(self, device):
        """Moves the Tensor to the given device"""
        self.choice_mat = self.choice_mat.to(device)
        self.choice_mask = self.choice_mask.to(device)
        self.sequence_mask = self.sequence_mask.to(device)
        return self


class NormBallTensor(BoundedTensor):
    """A tensor for which each is within some norm-ball of the original value."""

    def __init__(self, val, radius, p_norm):
        self.val = val
        self.radius = radius
        self.p_norm = p_norm


##### nn.Module's for BoundedTensor #####

class Linear(nn.Linear):
    """Linear layer."""

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return super(Linear, self).forward(x)
        if isinstance(x, IntervalBoundedTensor):
            z = F.linear(x.val, self.weight, self.bias)
            weight_abs = torch.abs(self.weight)
            mu_cur = (x.ub + x.lb) / 2
            r_cur = (x.ub - x.lb) / 2
            mu_new = F.linear(mu_cur, self.weight, self.bias)
            r_new = F.linear(r_cur, weight_abs)
            return IntervalBoundedTensor(z, mu_new - r_new, mu_new + r_new)
        elif isinstance(x, DiscreteChoiceTensorWithUNK):
            new_val = F.linear(x.val, self.weight, self.bias)
            new_choices = F.linear(x.choice_mat, self.weight, self.bias)
            return DiscreteChoiceTensorWithUNK(new_val, new_choices, x.choice_mask, x.sequence_mask, x.unk_mask)
        elif isinstance(x, DiscreteChoiceTensorTrans):
            new_choices = F.linear(x.choice_mat, self.weight, self.bias)
            return DiscreteChoiceTensorTrans(new_choices, x.choice_mask, x.sequence_mask)
        elif isinstance(x, NormBallTensor):
            q = 1.0 / (1.0 - 1.0 / x.p_norm)  # q from Holder's inequality
            z = F.linear(x.val, self.weight, self.bias)
            q_norm = torch.norm(self.weight, p=q, dim=1)  # Norm along in_dims axis
            delta = x.radius * q_norm
            return IntervalBoundedTensor(z, z - delta, z + delta)  # Broadcast out_dims
        else:
            raise TypeError(x)


class LinearOutput(Linear):
    """Linear output layer.

    A linear layer, but instead of computing interval bounds, computes

        max_{z feasible} c^T z + d

    where z is the output of this layer, for given vector(s) c and scalar(s) d.
    Following Gowal et al. (2018), we can get a slightly better bound here
    than by doing normal bound propagation.
    """

    def forward(self, x_ibp, c_list=None, d_list=None):
        """Compute linear output layer and bound on adversarial objective.

        Args:
          x_ibp: an ibp.Tensor of shape (batch_size, in_dims)
          c_list: list of torch.Tensor, each of shape (batch_size, out_dims)
          d_list: list of torch.Tensor, each of shape (batch_size,)
        Returns:
          x: ibp.Tensor of shape (batch_size, out_dims)
          bounds: if c_list and d_list, torch.Tensor of shape (batch_size,)
        """
        x, x_lb, x_ub = x_ibp
        z = F.linear(x, self.weight, self.bias)
        if c_list and d_list:
            bounds = []
            mu_cur = ((x_lb + x_ub) / 2).unsqueeze(1)  # B, 1, in_dims
            r_cur = ((x_ub - x_lb) / 2).unsqueeze(1)  # B, 1, in_dims
            for c, d in zip(c_list, d_list):
                c_prime = c.matmul(self.weight).unsqueeze(2)  # B, in_dims, 1
                d_prime = c.matmul(self.bias) + d  # B,
                c_prime_abs = torch.abs(c_prime)  # B, in_dims, 1
                mu_new = mu_cur.matmul(c_prime).view(-1)  # B,
                r_cur = r_cur.matmul(c_prime_abs).view(-1)  # B,
                bounds.append(mu_new + r_cur + d)
            return z, bounds
        else:
            return z


class Embedding(nn.Embedding):
    """nn.Embedding for DiscreteChoiceTensor.

    Note that unlike nn.Embedding, this module requires that the last dimension
    of the input is size 1, and will squeeze it before calling F.embedding.
    This requirement is due to how DiscreteChoiceTensor requires a dedicated
    dimension to represent the dimension along which values can change.
    """

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return super(Embedding, self).forward(x.squeeze(-1))
        if isinstance(x, DiscreteChoiceTensorWithUNK):
            if x.val.shape[-1] != 1:
                raise ValueError('Input tensor has shape %s, where last dimension != 1' % x.shape)
            new_val = F.embedding(
                x.val.squeeze(-1), self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
            new_choices = F.embedding(
                x.choice_mat.squeeze(-1), self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
            return DiscreteChoiceTensorWithUNK(new_val, new_choices, x.choice_mask, x.sequence_mask, x.unk_mask)
        elif isinstance(x, DiscreteChoiceTensorTrans):
            new_choices = F.embedding(
                x.choice_mat.squeeze(-1), self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
            return DiscreteChoiceTensorTrans(new_choices, x.choice_mask, x.sequence_mask)
        else:
            raise TypeError(x)


class Conv1d(nn.Conv1d):
    """One-dimensional convolutional layer.

    Works the same as a linear layer.
    """

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return super(Conv1d, self).forward(x)
        if isinstance(x, IntervalBoundedTensor):
            z = F.conv1d(x.val, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
            weight_abs = torch.abs(self.weight)
            mu_cur = (x.ub + x.lb) / 2
            r_cur = (x.ub - x.lb) / 2
            mu_new = F.conv1d(mu_cur, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
            r_new = F.conv1d(r_cur, weight_abs, None, self.stride,
                             self.padding, self.dilation, self.groups)
            return IntervalBoundedTensor(z, mu_new - r_new, mu_new + r_new)
        else:
            raise TypeError(x)


class MaxPool1d(nn.MaxPool1d):
    """One-dimensional max-pooling layer."""

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return super(MaxPool1d, self).forward(x)
        elif isinstance(x, IntervalBoundedTensor):
            z = F.max_pool1d(x.val, self.kernel_size, self.stride, self.padding,
                             self.dilation, self.ceil_mode, self.return_indices)
            lb = F.max_pool1d(x.lb, self.kernel_size, self.stride, self.padding,
                              self.dilation, self.ceil_mode, self.return_indices)
            ub = F.max_pool1d(x.ub, self.kernel_size, self.stride, self.padding,
                              self.dilation, self.ceil_mode, self.return_indices)
            return IntervalBoundedTensor(z, lb, ub)
        else:
            raise TypeError(x)


class LSTM(nn.Module):
    """An LSTM."""

    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.i2h = Linear(input_size, 4 * hidden_size)
        self.h2h = Linear(hidden_size, 4 * hidden_size)
        if bidirectional:
            self.back_i2h = Linear(input_size, 4 * hidden_size)
            self.back_h2h = Linear(hidden_size, 4 * hidden_size)

    def _step(self, h, c, x_t, i2h, h2h, analysis_mode=False):
        preact = add(i2h(x_t), h2h(h))
        g_t = activation(torch.tanh, preact[:, 3 * self.hidden_size:])
        gates = activation(torch.sigmoid, preact[:, :3 * self.hidden_size])
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, 2 * self.hidden_size:]
        c_t = add(mul(c, f_t), mul(i_t, g_t))
        h_t = mul(o_t, activation(torch.tanh, c_t))
        if analysis_mode:
            return h_t, c_t, i_t, f_t, o_t
        return h_t, c_t

    def _process(self, h, c, x, i2h, h2h, reverse=False, mask=None, analysis_mode=False):
        B, T, d = x.shape  # batch_first=True
        idxs = range(T)
        if reverse:
            idxs = idxs[::-1]
        h_seq = []
        c_seq = []
        if analysis_mode:
            i_seq = []
            f_seq = []
            o_seq = []
        for i in idxs:
            x_t = x[:, i, :]  # B, d_in
            if analysis_mode:
                h_t, c_t, i_t, f_t, o_t = self._step(h, c, x_t, i2h, h2h, analysis_mode=True)
                i_seq.append(i_t)
                f_seq.append(f_t)
                o_seq.append(o_t)
            else:
                h_t, c_t = self._step(h, c, x_t, i2h, h2h)
            if mask is not None:
                # Don't update h or c when mask is 0
                mask_t = mask[:, i].unsqueeze(1)  # B,1
                h = h_t * mask_t + h * (1.0 - mask_t)
                c = c_t * mask_t + c * (1.0 - mask_t)
            h_seq.append(h)
            c_seq.append(c)
        if reverse:
            h_seq = h_seq[::-1]
            c_seq = c_seq[::-1]
            if analysis_mode:
                i_seq = i_seq[::-1]
                f_seq = f_seq[::-1]
                o_seq = o_seq[::-1]
        if analysis_mode:
            return h_seq, c_seq, i_seq, f_seq, o_seq
        return h_seq, c_seq

    def forward(self, x, s0, mask=None, analysis_mode=False):
        """Forward pass of LSTM

        Args:
          x: word vectors, size (B, T, d)
          s0: tuple of (h0, x0) where each is (B, d), or (B, 2d) if bidirectional=True
          mask: If provided, 0-1 mask of size (B, T)
        """
        h0, c0 = s0  # Each is (B, d), or (B, 2d) if bidirectional=True
        if self.bidirectional:
            h0_back = h0[:, self.hidden_size:]
            h0 = h0[:, :self.hidden_size]
            c0_back = c0[:, self.hidden_size:]
            c0 = c0[:, :self.hidden_size]
        if analysis_mode:
            h_seq, c_seq, i_seq, f_seq, o_seq = self._process(
                h0, c0, x, self.i2h, self.h2h, mask=mask, analysis_mode=True)
        else:
            h_seq, c_seq = self._process(h0, c0, x, self.i2h, self.h2h, mask=mask)
        if self.bidirectional:
            if analysis_mode:
                h_back_seq, c_back_seq, i_back_seq, f_back_seq, o_back_seq = self._process(
                    h0_back, c0_back, x, self.back_i2h, self.back_h2h, reverse=True, mask=mask,
                    analysis_mode=True)
                i_seq = [cat((f, b), dim=1) for f, b in zip(i_seq, i_back_seq)]
                f_seq = [cat((f, b), dim=1) for f, b in zip(f_seq, f_back_seq)]
                o_seq = [cat((f, b), dim=1) for f, b in zip(o_seq, o_back_seq)]
            else:
                h_back_seq, c_back_seq = self._process(
                    h0_back, c0_back, x, self.back_i2h, self.back_h2h, reverse=True, mask=mask)
            h_seq = [cat((hf, hb), dim=1) for hf, hb in zip(h_seq, h_back_seq)]
            c_seq = [cat((cf, cb), dim=1) for cf, cb in zip(c_seq, c_back_seq)]
        h_mat = stack(h_seq, dim=1)  # list of (B, d) -> (B, T, d)
        c_mat = stack(c_seq, dim=1)  # list of (B, d) -> (B, T, d)
        if analysis_mode:
            i_mat = stack(i_seq, dim=1)
            f_mat = stack(f_seq, dim=1)
            o_mat = stack(o_seq, dim=1)
            return h_mat, c_mat, (i_mat, f_mat, o_mat)
        return h_mat, c_mat


class LSTMDP(nn.Module):
    """An LSTM."""

    def __init__(self, input_size, hidden_size, perturbation, bidirectional=False, baseline=False):
        super(LSTMDP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.perturbation = perturbation
        self.baseline = baseline
        self.bidirectional = bidirectional
        self.i2h = Linear(input_size, 4 * hidden_size)
        self.h2h = Linear(hidden_size, 4 * hidden_size)
        self.deltas = Perturbation.str2deltas(perturbation)
        self.deltas_p1 = [delta + 1 for delta in self.deltas]
        self.Ins_delta = self.deltas[1]
        if bidirectional:
            self.back_i2h = Linear(input_size, 4 * hidden_size)
            self.back_h2h = Linear(hidden_size, 4 * hidden_size)

    def _step(self, h, c, x_t, i2h, h2h, analysis_mode=False):
        preact = add(i2h(x_t), h2h(h))
        g_t = activation(torch.tanh, preact[:, 3 * self.hidden_size:])
        gates = activation(torch.sigmoid, preact[:, :3 * self.hidden_size])
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, 2 * self.hidden_size:]
        c_t = add(mul(c, f_t), mul(i_t, g_t))
        h_t = mul(o_t, activation(torch.tanh, c_t))
        if analysis_mode:
            return h_t, c_t, i_t, f_t, o_t
        return h_t, c_t

    def _process(self, h, c, x, i2h, h2h, reverse=False, mask=None, analysis_mode=False):
        B, T, d = x.shape  # batch_first=True
        idxs = range(T)
        if reverse:
            idxs = idxs[::-1]
        h_seq = []
        c_seq = []
        if analysis_mode:
            i_seq = []
            f_seq = []
            o_seq = []
        for i in idxs:
            x_t = x[:, i, :]  # B, d_in
            if analysis_mode:
                h_t, c_t, i_t, f_t, o_t = self._step(h, c, x_t, i2h, h2h, analysis_mode=True)
                i_seq.append(i_t)
                f_seq.append(f_t)
                o_seq.append(o_t)
            else:
                h_t, c_t = self._step(h, c, x_t, i2h, h2h)
            if mask is not None:
                # Don't update h or c when mask is 0
                mask_t = mask[:, i].unsqueeze(1)  # B,1
                h = h_t * mask_t + h * (1.0 - mask_t)
                c = c_t * mask_t + c * (1.0 - mask_t)
            h_seq.append(h)
            c_seq.append(c)
        if analysis_mode:
            return h_seq, c_seq, i_seq, f_seq, o_seq
        return h_seq, c_seq

    def _processDP(self, h, c, x, i2h, h2h, reverse=False, mask=None, analysis_mode=False, output=None, unk_mask=None):
        B, T, _ = x.shape  # batch_first=True
        d = self.hidden_size
        D = len(self.deltas)
        idxs = list(range(T))
        if reverse:
            if self.Ins_delta > 0:
                idxs = idxs[:-self.Ins_delta][::-1] + idxs[-self.Ins_delta:]  # we keep the paddings at the end
            else:
                idxs = idxs[::-1]
        x = IntervalBoundedTensor.point(x)  # make x: Tensor as a IntervalBoundedTensor

        def compute_state(h, c, x_t, mask_t, unk_mask_t):
            if not self.baseline:
                # x_t, mask_t, unk_mask_t can be longer than h, we need to truncate them
                x_t = x_t[:h.shape[0]]
                mask_t = mask_t[:h.shape[0]] if mask_t is not None else None
                unk_mask_t = unk_mask_t[:h.shape[0]] if unk_mask_t is not None else None

            if analysis_mode:
                h_t, c_t, i_t, f_t, o_t = self._step(h, c, x_t, i2h, h2h, analysis_mode=True)
            else:
                h_t, c_t = self._step(h, c, x_t, i2h, h2h)
            if mask_t is not None:
                # Don't update h or c when mask is 0
                h_t = h_t * mask_t + h * (1.0 - mask_t)
                c_t = c_t * mask_t + c * (1.0 - mask_t)
            if unk_mask_t is not None:
                merge_h = h_t.merge(h)
                # unk_mask_t[i] = 1 then h_t == (old) h_t, otherwise h_t = merge_h, i.e.,
                # we merge the state when mask is 0
                merge_c = c_t.merge(c)
                h_t = h_t * unk_mask_t + merge_h * (1.0 - unk_mask_t)
                c_t = c_t * unk_mask_t + merge_c * (1.0 - unk_mask_t)

            if analysis_mode:
                return h_t, c_t, i_t, f_t, o_t
            return h_t, c_t

        def dup(x):
            return x.view([1] * D + [B, -1]).repeat(*self.deltas_p1, 1, 1)

        ans = []
        if self.baseline:
            for i in idxs:
                post_state = compute_state(h, c, output[:, i, :],
                                           mask[:, i].unsqueeze(-1) if mask is not None else None,
                                           unk_mask[:, i].unsqueeze(-1) if unk_mask is not None else None)

                ans.append(post_state)
                h, c = post_state[0], post_state[1]
        else:
            h = dup(h)
            c = dup(c)
            x = x.repeat(np.prod(self.deltas_p1), 1, 1)
            output = output.repeat(np.prod(self.deltas_p1), 1, 1)
            mask = mask.repeat(np.prod(self.deltas_p1), 1) if mask is not None else None
            unk_mask = unk_mask.repeat(np.prod(self.deltas_p1), 1) if unk_mask is not None else None
            for idxs_idx, i in enumerate(idxs):
                # identity
                del_extend = None
                ins_extend = None
                sub_extend = None
                # Del
                if self.deltas[0] > 0:  # Del, [here,:,:,:,:] (Del, Ins, Sub, B, d)
                    def del_ins_j(h, c, x, mask, unk_mask):
                        del_1 = compute_state(h[:-1, :, :, :, :].reshape(-1, d), c[:-1, :, :, :, :].reshape(-1, d)
                                              , x, mask, unk_mask)

                        del_00 = compute_state(h[:1, :, :, :, :].reshape(-1, d), c[:1, :, :, :, :].reshape(-1, d)
                                               , x, mask, None)
                        del_00_extend = view(del_00, 1, 1, self.deltas[2] + 1, B, d)

                        del_01 = compute_state(h[1:, :, :, :, :].reshape(-1, d), c[1:, :, :, :, :].reshape(-1, d)
                                               , x, mask, None)
                        del_extend = view(merge(del_1, del_01), self.deltas[0], 1, self.deltas[2] + 1, B, d)
                        del_extend = tuple([cat([s, t], dim=0) for s, t in zip(del_00_extend, del_extend)])
                        return del_extend

                    del_extend = del_ins_j(h[:, :1, :, :, :], c[:, :1, :, :, :], x[:, i, :],
                                           mask[:, i].unsqueeze(-1) if mask is not None else None,
                                           unk_mask[:, i].unsqueeze(-1) if unk_mask is not None else None)

                    for j in range(1, self.deltas_p1[1]):
                        if idxs_idx >= j * 2:
                            del_extend_t = del_ins_j(h[:, j:j + 1, :, :, :], c[:, j:j + 1, :, :, :],
                                                     x[:, idxs[idxs_idx - j], :],
                                                     mask[:, idxs[idxs_idx - j]].unsqueeze(
                                                         -1) if mask is not None else None,
                                                     unk_mask[:, idxs[idxs_idx - j]].unsqueeze(
                                                         -1) if unk_mask is not None else None)
                            del_extend = tuple([cat([s, t], dim=1) for s, t in zip(del_extend, del_extend_t)])
                        else:
                            del_extend = tuple([cat([s, s[:, :1, :, :, :]], dim=1) for s in del_extend])

                # Ins
                if self.deltas[1] > 0:  # Ins, [:,here,:,:,:] (Del, Ins, Sub, B, d)
                    # origin input
                    ins_extend = None
                    ins_extend_special = None
                    for j in range(self.deltas_p1[1]):
                        # Case 1.1: f[i-1][j] --> f[i][j], first time
                        if idxs_idx >= j * 2:
                            ins_extend_t = compute_state(h[:, j, :, :, :].reshape(-1, d),
                                                         c[:, j, :, :, :].reshape(-1, d),
                                                         x[:, idxs[idxs_idx - j], :],
                                                         mask[:, idxs[idxs_idx - j]].unsqueeze(
                                                             -1) if mask is not None else None,
                                                         None)
                        else:
                            ins_extend_t = ins_extend[0][:, :1, :, :, :].reshape(-1, d), \
                                           ins_extend[1][:, :1, :, :, :].reshape(-1, d)
                        # Case 1.2: f_special[i-1][j-1] --> f[i][j], second time
                        if j > 0 and ins_extend_special_pre is not None and idxs_idx >= j * 2 - 1:
                            ins_extend_t_1 = compute_state(ins_extend_special_pre[0][:, j - 1, :, :, :].reshape(-1, d),
                                                           ins_extend_special_pre[1][:, j - 1, :, :, :].reshape(-1, d),
                                                           x[:, idxs[idxs_idx - j], :],
                                                           mask[:, idxs[idxs_idx - j]].unsqueeze(
                                                               -1) if mask is not None else None,
                                                           None)
                            ins_extend_t = merge(ins_extend_t, ins_extend_t_1)
                        # Case 1.3: f[i-1][j] -> f_special[i][j], first time
                        if idxs_idx >= j * 2 and j < self.deltas_p1[1]:
                            ins_extend_t_special = compute_state(h[:, j, :, :, :].reshape(-1, d),
                                                                 c[:, j, :, :, :].reshape(-1, d),
                                                                 x[:, idxs[idxs_idx - j], :],
                                                                 mask[:, idxs[idxs_idx - j]].unsqueeze(
                                                                     -1) if mask is not None else None,
                                                                 None)
                        else:
                            ins_extend_t_special = ins_extend_special[0][:, :1, :, :, :].reshape(-1, d), \
                                                   ins_extend_special[1][:, :1, :, :, :].reshape(-1, d)
                        ins_extend_t = view(ins_extend_t, self.deltas[0] + 1, 1, self.deltas[2] + 1, B, d)
                        ins_extend_t_special = view(ins_extend_t_special, self.deltas[0] + 1, 1, self.deltas[2] + 1, B,
                                                    d)
                        if ins_extend is None:
                            ins_extend = ins_extend_t
                            ins_extend_special = ins_extend_t_special
                        else:
                            ins_extend = tuple([cat([s, t], dim=1) for s, t in zip(ins_extend, ins_extend_t)])
                            ins_extend_special = tuple(
                                [cat([s, t], dim=1) for s, t in zip(ins_extend_special, ins_extend_t_special)])

                    ins_extend_special_pre = ins_extend_special

                if self.deltas[2] > 0:  # Sub, [:,:,here,:,:] (Del, Ins, Sub, B, d)
                    def sub_ins_j(h, c, x, output, mask):
                        sub_1 = compute_state(h[:, :, :-1, :, :].reshape(-1, d), c[:, :, :-1, :, :].reshape(-1, d),
                                              output, mask, None)

                        sub_00 = compute_state(h[:, :, :1, :, :].reshape(-1, d), c[:, :, :1, :, :].reshape(-1, d), x,
                                               mask, None)
                        sub_00_extend = view(sub_00, self.deltas[0] + 1, 1, 1, B, d)

                        sub_01 = compute_state(h[:, :, 1:, :, :].reshape(-1, d), c[:, :, 1:, :, :].reshape(-1, d), x,
                                               mask, None)
                        sub_extend = view(merge(sub_1, sub_01), self.deltas[0] + 1, 1, self.deltas[2], B, d)
                        sub_extend = tuple([cat([s, t], dim=2) for s, t in zip(sub_00_extend, sub_extend)])
                        return sub_extend

                    sub_extend = sub_ins_j(h[:, :1, :, :, :], c[:, :1, :, :, :], x[:, i, :], output[:, i, :],
                                           mask[:, i].unsqueeze(-1) if mask is not None else None)

                    for j in range(1, self.deltas_p1[1]):
                        if idxs_idx >= j * 2:
                            sub_extend_t = sub_ins_j(h[:, j:j + 1, :, :, :], c[:, j:j + 1, :, :, :],
                                                     x[:, idxs[idxs_idx - j], :], output[:, idxs[idxs_idx - j], :],
                                                     mask[:, idxs[idxs_idx - j]].unsqueeze(
                                                         -1) if mask is not None else None)
                            sub_extend = tuple([cat([s, t], dim=1) for s, t in zip(sub_extend, sub_extend_t)])
                        else:
                            sub_extend = tuple([cat([s, s[:, :1, :, :, :]], dim=1) for s in sub_extend])

                extends = [del_extend, ins_extend, sub_extend]
                extend = None
                for tran_extend in extends:
                    if extend is None:
                        extend = tran_extend
                    elif tran_extend is not None:
                        extend = merge(extend, tran_extend)

                assert extend is not None
                h, c = extend[0], extend[1]
                # print((h.ub - h.lb).sum())

                post_state = view(extend, -1, d)  # [-1, d]
                ans_t = tuple([t[:B, :] for t in post_state])
                for j in range(B, post_state[0].shape[0], B):
                    ans_t = tuple([t.merge(t_[j:j + B]) for t, t_ in zip(ans_t, post_state)])
                ans.append(ans_t)

        ret = [[] for _ in range(len(ans[0]))]
        for i in range(len(ans)):
            for j in range(len(ret)):
                ret[j].append(ans[i][j])

        return ret

    def forward(self, x, x_interval, s0, mask=None, analysis_mode=False, unk_mask=None):
        """Forward pass of LSTM

        Args:
          x: word vectors, size (B, T, d)
          s0: tuple of (h0, x0) where each is (B, d), or (B, 2d) if bidirectional=True
          mask: If provided, 0-1 mask of size (B, T)
        """
        h0, c0 = s0  # Each is (B, d), or (B, 2d) if bidirectional=True
        if self.bidirectional:
            h0_back = h0[:, self.hidden_size:]
            h0 = h0[:, :self.hidden_size]
            c0_back = c0[:, self.hidden_size:]
            c0 = c0[:, :self.hidden_size]
        if x_interval is None:
            process = self._process
        else:
            process = partial(self._processDP, output=x_interval, unk_mask=unk_mask)
        if analysis_mode:
            h_seq, c_seq, i_seq, f_seq, o_seq = process(
                h0, c0, x, self.i2h, self.h2h, mask=mask, analysis_mode=True)
        else:
            h_seq, c_seq = process(h0, c0, x, self.i2h, self.h2h, mask=mask)
        if self.bidirectional:
            if analysis_mode:
                h_back_seq, c_back_seq, i_back_seq, f_back_seq, o_back_seq = process(
                    h0_back, c0_back, x, self.back_i2h, self.back_h2h, reverse=True, mask=mask,
                    analysis_mode=True)
                i_seq = [cat((f, b), dim=1) for f, b in zip(i_seq, i_back_seq)]
                f_seq = [cat((f, b), dim=1) for f, b in zip(f_seq, f_back_seq)]
                o_seq = [cat((f, b), dim=1) for f, b in zip(o_seq, o_back_seq)]
            else:
                h_back_seq, c_back_seq = process(
                    h0_back, c0_back, x, self.back_i2h, self.back_h2h, reverse=True, mask=mask)
            h_seq = [cat((hf, hb), dim=1) for hf, hb in zip(h_seq, h_back_seq)]
            c_seq = [cat((cf, cb), dim=1) for cf, cb in zip(c_seq, c_back_seq)]
        h_mat = stack(h_seq, dim=1)  # list of (B, d) -> (B, T, d)
        c_mat = stack(c_seq, dim=1)  # list of (B, d) -> (B, T, d)
        if analysis_mode:
            i_mat = stack(i_seq, dim=1)
            f_mat = stack(f_seq, dim=1)
            o_mat = stack(o_seq, dim=1)
            return h_mat, c_mat, (i_mat, f_mat, o_mat)
        return h_mat, c_mat


class LSTMDPGeneral(nn.Module):
    """An LSTM."""

    def __init__(self, input_size, hidden_size, perturbation, device):
        super(LSTMDPGeneral, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.perturbation, self.deltas = perturbation
        self.i2h = Linear(input_size, 4 * hidden_size)
        self.h2h = Linear(hidden_size, 4 * hidden_size)
        self.deltas_p1 = [delta + 1 for delta in self.deltas]
        self.deltas_ranges = [range(x) for x in self.deltas_p1]

    def _step(self, h, c, x_t, i2h, h2h, analysis_mode=False):
        preact = add(i2h(x_t), h2h(h))
        g_t = activation(torch.tanh, preact[:, 3 * self.hidden_size:])
        gates = activation(torch.sigmoid, preact[:, :3 * self.hidden_size])
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, 2 * self.hidden_size:]
        c_t = add(mul(c, f_t), mul(i_t, g_t))
        h_t = mul(o_t, activation(torch.tanh, c_t))
        if analysis_mode:
            return h_t, c_t, i_t, f_t, o_t
        return h_t, c_t

    def _process(self, h, c, x, i2h, h2h, length, mask=None, analysis_mode=False):

        B, T, d = x.shape  # batch_first=True
        idxs = range(T)
        h_seq = []
        c_seq = []
        if analysis_mode:
            i_seq = []
            f_seq = []
            o_seq = []
        for i in idxs:
            x_t = x[:, i, :]  # B, d_in
            if analysis_mode:
                h_t, c_t, i_t, f_t, o_t = self._step(h, c, x_t, i2h, h2h, analysis_mode=True)
                i_seq.append(i_t)
                f_seq.append(f_t)
                o_seq.append(o_t)
            else:
                h_t, c_t = self._step(h, c, x_t, i2h, h2h)
            if mask is not None:
                # Don't update h or c when mask is 0
                mask_t = mask[:, i].unsqueeze(1)  # B,1
                h = h_t * mask_t + h * (1.0 - mask_t)
                c = c_t * mask_t + c * (1.0 - mask_t)
            h_seq.append(h)
            c_seq.append(c)

        if analysis_mode:
            return h_seq, c_seq, i_seq, f_seq, o_seq, length
        return h_seq, c_seq, length

    def _processDP(self, _trans_output: tuple, h, c, x, i2h, h2h, length, mask=None):
        """
        A general DP
        :param _trans_output: a tuple (trans_o, trans_phi)
        trans_o is a list of interval bound tensors, each of which has the shape (B, T, tran.t, word_vec_size / h_size).
        trans_o[tran_id][:, i, j, :] means that trans[tran_id] matches at the input positions
        i ~ (i + trans[tran_id].s - 1), how the j-th output of trans[id] can be. (j = 0 ~ trans[tran_id].t - 1).
        trans_phi is a list of bool tensors, each of which has the shape (B, T).
        :param h: hidden states
        :param c: cell states
        :param x: input sequence (B, T, word_vec_size or h_size)
        :param i2h: input to hidden linear layer
        :param h2h: hidden to hidden linear layer
        :param mask: mask for the input sequence (B, T)
        :return: a list of hidden states in the interval bound form and a possible length interval
        """
        global DEBUG
        trans_o, trans_phi = _trans_output
        B, T, _ = x.shape  # batch_first=True
        max_out_len = T
        for delta, tran in zip(self.deltas, self.perturbation):
            if tran.t > tran.s:
                max_out_len += (tran.t - tran.s) * delta
        mask = mask.unsqueeze(-1)
        d = self.hidden_size
        D = len(self.deltas)
        x = IntervalBoundedTensor.point(x)  # make x: Tensor as a IntervalBoundedTensor
        length_lb = [-1] * B
        length_ub = [-1] * B

        def compute_state(h, c, x_t, mask_t):
            h_t, c_t = self._step(h, c, x_t, i2h, h2h)
            if mask_t is not None:
                # Don't update h or c when mask is 0
                h_t = h_t * mask_t + h * (1.0 - mask_t)
                c_t = c_t * mask_t + c * (1.0 - mask_t)

            return h_t, c_t

        def cal_in_pos(out_pos, deltas):
            # compute the input position for deltas
            in_pos = out_pos
            for delta, tran in zip(deltas, self.perturbation):
                in_pos += (tran.s - tran.t) * delta
            return in_pos

        def mask_by_feasible(feasible_mask, overapp_cur_states, defaults):
            DEBUG = True
            return (where(feasible_mask.unsqueeze(-1), overapp_cur_states[0], defaults[0]),
                    where(feasible_mask.unsqueeze(-1), overapp_cur_states[1], defaults[1]))

        # a feasible set of states (max_out_len + 1, B, *self.deltas_p1)
        feasible = torch.zeros(max_out_len + 1, B, *self.deltas_p1).bool().to(self.device)
        all_zeros = (0,) * len(self.deltas_p1)
        # Caution! A dangerous assumption: we use feasible[-1] to denote the actual -1 index instead of the last one,
        # What's why we make the length of feasible max_out_len + 1
        feasible[(-1, slice(None),) + all_zeros] = 1
        ans = []
        len_cur_states = np.prod(self.deltas_p1)
        state2id = {}
        for state_id, deltas in enumerate(product(*self.deltas_ranges)):
            state2id[deltas] = state_id
        cur_states = [(IntervalBoundedTensor.point(h), IntervalBoundedTensor.point(c))] + [None] * (len_cur_states - 1)
        bottom_h_size_like = IntervalBoundedTensor.bottom_like(h)
        for i in range(max_out_len + 1):  # + 1 make sure the last state is added.
            # cur_states should read as pre_states here.
            for state_id, deltas in enumerate(product(*self.deltas_ranges)):
                # compute the input position for deltas
                _in_pos = cal_in_pos(i - 1, deltas)

                for tran_id, tran in enumerate(self.perturbation):
                    if deltas[tran_id] > 0 and tran.t == 0:
                        pre_deltas = deltas[:tran_id] + (deltas[tran_id] - 1,) + deltas[tran_id + 1:]
                        # the input position for pre_deltas
                        in_pos = _in_pos - (tran.s - tran.t) + 1  # out_pos is the pos previous start_pos
                        if 0 <= in_pos < T:
                            feasible_mask = feasible[(i - 1, slice(None),) + pre_deltas] & trans_phi[tran_id][:,
                                                                                           in_pos] & (in_pos < length)
                            if feasible_mask.any().item():
                                merge_cur_states = merge(cur_states[state_id], cur_states[state2id[pre_deltas]])
                                cur_states[state_id] = mask_by_feasible(feasible_mask, merge_cur_states,
                                                                        cur_states[state_id])
                                feasible[(i - 1, slice(None),) + deltas] |= feasible_mask

            # update ans and h, c
            cur_states_ret = [[], []]
            cur_ans = None
            for cur_state in cur_states:
                cur_ans = merge(cur_ans, cur_state)
                for j in range(len(cur_states_ret)):
                    cur_states_ret[j].append(cur_state[j] if cur_state is not None else bottom_h_size_like)

            for j in range(len(cur_states_ret)):
                cur_states_ret[j] = stack(cur_states_ret[j], dim=0)

            h, c = cur_states_ret[0].view(*self.deltas_p1, B, d), cur_states_ret[1].view(*self.deltas_p1, B, d)
            if i > 0:
                ans.append(cur_ans)
                # calculate the possible length of each sentence, the length_lb and ub are not related to the NN
                # parameters. In other words, it is ok to decouple them from the gradient descent process.
                for s_id in range(B):
                    for deltas in product(*self.deltas_ranges):
                        in_pos = cal_in_pos(i - 1, deltas)
                        if in_pos == length[s_id].item() - 1 and feasible[(i - 1, s_id,) + deltas].item():
                            length_ub[s_id] = i
                            if length_lb[s_id] == -1:
                                length_lb[s_id] = i

            # calculate cur_states by DP
            cur_states = []
            for deltas in product(*self.deltas_ranges):
                _in_pos = cal_in_pos(i, deltas)

                # Case 1: do not transform at this position
                cur_state = None
                # _in_pos cannot exceed T
                feasible_mask = feasible[(i - 1, slice(None),) + deltas] & (_in_pos < length)
                if feasible_mask.any().item() and 0 <= _in_pos < T:
                    cur_state = mask_by_feasible(feasible_mask,
                                                 compute_state(h[deltas], c[deltas], x[:, _in_pos, :],
                                                               mask[:, _in_pos, :]),
                                                 (bottom_h_size_like, bottom_h_size_like))
                    feasible[(i, slice(None),) + deltas] = feasible_mask

                # Case 2: let's enumerate a transformation
                # Case 2-1: if it is inside a transformation, the j in [0, tran.t - 1)
                for tran_id, tran in enumerate(self.perturbation):
                    if deltas[tran_id] < self.deltas[tran_id] and tran.t > 0:
                        pre_deltas = deltas
                        # the input position for pre_deltas
                        in_pos = _in_pos - (tran.s - tran.t)
                        for j in range(tran.t - 1):
                            # if in_pos - j does not exceed T and there is any sentence matching at in_pos - j
                            if 0 <= in_pos - j < T:
                                feasible_mask = (in_pos - j < length) & feasible[
                                    (i - 1 - j, slice(None),) + pre_deltas] & trans_phi[tran_id][:, in_pos - j]
                                if feasible_mask.any().item():
                                    DEBUG = False
                                    masked_cur_state = mask_by_feasible(feasible_mask,
                                                                        compute_state(h[pre_deltas], c[pre_deltas],
                                                                                      trans_o[tran_id][:,
                                                                                      in_pos - j, j, :], None),
                                                                        (bottom_h_size_like, bottom_h_size_like))
                                    cur_state = merge(cur_state, masked_cur_state)

                # Case 2-2: if it is at the end of a transformation, the j == tran.t - 1
                for tran_id, tran in enumerate(self.perturbation):
                    if deltas[tran_id] > 0 and tran.t > 0:
                        pre_deltas = deltas[:tran_id] + (deltas[tran_id] - 1,) + deltas[tran_id + 1:]
                        # the input position for pre_deltas
                        in_pos = _in_pos - (tran.s - tran.t)
                        j = tran.t - 1
                        # if in_pos - j does not exceed T and there is any sentence matching at in_pos - j
                        if 0 <= in_pos - j < T:
                            feasible_mask = (in_pos - j < length) & feasible[
                                (i - 1 - j, slice(None),) + pre_deltas] & trans_phi[tran_id][:, in_pos - j]
                            if feasible_mask.any().item():
                                DEBUG = False
                                masked_cur_state = mask_by_feasible(feasible_mask,
                                                                    compute_state(h[pre_deltas], c[pre_deltas],
                                                                                  trans_o[tran_id][:,
                                                                                  in_pos - j, j, :], None),
                                                                    (bottom_h_size_like, bottom_h_size_like))
                                cur_state = merge(cur_state, masked_cur_state)
                                feasible[(i, slice(None),) + deltas] |= feasible_mask

                cur_states.append(cur_state)

            # if feasible is not updated, then the DP process is over.
            # some may doubt that we need to break after considering tran.t == 0, but the longest perturbed sentences
            # cannot take any such transformation.
            if not feasible[i].any().item():
                break

        ret = [[] for _ in range(len(ans[0]))]
        for i in range(len(ans)):
            for j in range(len(ret)):
                ret[j].append(ans[i][j])

        assert min(length_lb) > 0 and min(length_ub) > 0 and max(length_ub) == len(ret[0])
        return ret + [IntervalBoundedTensor(length, torch.Tensor(length_lb).long().to(self.device),
                                            torch.Tensor(length_ub).long().to(self.device))]

    def forward(self, x, trans_output, s0, length, mask=None):
        """Forward pass of LSTM

        Args:
          x: word vectors, size (B, T, d)
          s0: tuple of (h0, x0) where each is (B, d)
          mask: If provided, 0-1 mask of size (B, T)
        """
        h0, c0 = s0  # Each is (B, d)
        if trans_output is None:
            process = self._process
        else:
            process = partial(self._processDP, trans_output)
        h_seq, c_seq, length_interval = process(h0, c0, x, self.i2h, self.h2h, length, mask=mask)

        h_mat = stack(h_seq, dim=1)  # list of (B, d) -> (B, T, d)
        c_mat = stack(c_seq, dim=1)  # list of (B, d) -> (B, T, d)
        return h_mat, c_mat, length_interval


class GRU(nn.Module):
    """A GRU."""

    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.i2h = Linear(input_size, 3 * hidden_size)
        self.h2h = Linear(hidden_size, 3 * hidden_size)
        if bidirectional:
            self.back_i2h = Linear(input_size, 3 * hidden_size)
            self.back_h2h = Linear(hidden_size, 3 * hidden_size)

    def _step(self, h, x_t, i2h, h2h):
        i_out = i2h(x_t)
        h_out = h2h(h)
        preact = add(i_out[:, :2 * self.hidden_size], h_out[:, :2 * self.hidden_size])
        gates = activation(torch.sigmoid, preact)
        r_t = gates[:, :self.hidden_size]
        z_t = gates[:, self.hidden_size:]
        i_state = i_out[:, 2 * self.hidden_size:]
        h_state = h_out[:, 2 * self.hidden_size:]
        n_t = activation(torch.tanh, i_state + mul(r_t, h_state))
        if isinstance(z_t, torch.Tensor):
            ones = torch.ones_like(z_t)
        else:
            ones = torch.ones_like(z_t.val)
        h_t = add(mul(add(ones, - z_t), n_t), mul(z_t, h))
        return h_t

    def _process(self, h, x, i2h, h2h, reverse=False, mask=None):
        B, T, d = x.shape  # batch_first=True
        idxs = range(T)
        if reverse:
            idxs = idxs[::-1]
        h_seq = []
        for i in idxs:
            x_t = x[:, i, :]  # B, d_in
            h_t = self._step(h, x_t, i2h, h2h)
            if mask is not None:
                # Don't update h when mask is 0
                mask_t = mask[:, i].unsqueeze(1)  # B,1
                h = h_t * mask_t + h * (1.0 - mask_t)
            h_seq.append(h)
        if reverse:
            h_seq = h_seq[::-1]
        return h_seq

    def forward(self, x, h0, mask=None):
        """Forward pass of GRU

        Args:
          x: word vectors, size (B, T, d)
          h0: tuple of (h0, x0) where each is (B, d), or (B, 2d) if bidirectional=True
          mask: If provided, 0-1 mask of size (B, T)
        """
        if self.bidirectional:
            h0_back = h0[:, self.hidden_size:]
            h0 = h0[:, :self.hidden_size]
        h_seq = self._process(h0, x, self.i2h, self.h2h, mask=mask)
        if self.bidirectional:
            h_back_seq = self._process(
                h0_back, x, self.back_i2h, self.back_h2h, reverse=True, mask=mask)
            h_seq = [cat((hf, hb), dim=1) for hf, hb in zip(h_seq, h_back_seq)]
        h_mat = stack(h_seq, dim=1)  # list of (B, d) -> (B, T, d)
        return h_mat


class Dropout(nn.Dropout):
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return super(Dropout, self).forward(x)
        elif isinstance(x, IntervalBoundedTensor):
            if self.training:
                probs = torch.full_like(x.val, 1.0 - self.p)
                mask = torch.distributions.Bernoulli(probs).sample() / (1.0 - self.p)
                return IntervalBoundedTensor(mask * x.val, mask * x.lb, mask * x.ub)
            else:
                return x
        else:
            raise TypeError(x)


def add(x1, x2):
    """Sum two tensors."""
    # I think we have to do it this way and not as operator overloading,
    # to catch the case of torch.Tensor.__add__(IntervalBoundedTensor)
    if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
        return x1 + x2
    elif isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
        if isinstance(x2, torch.Tensor):
            x1, x2 = x2, x1  # WLOG x1 is torch.Tensor
        if isinstance(x2, IntervalBoundedTensor):
            return IntervalBoundedTensor(x2.val + x1, x2.lb + x1, x2.ub + x1)
        else:
            raise TypeError(x1, x2)
    else:
        if isinstance(x1, IntervalBoundedTensor) and isinstance(x2, IntervalBoundedTensor):
            return IntervalBoundedTensor(x1.val + x2.val, x1.lb + x2.lb, x1.ub + x2.ub)
        else:
            raise TypeError(x1, x2)


def mul(x1, x2):
    """Elementwise multiplication of two tensors."""
    if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
        return torch.mul(x1, x2)
    elif isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
        if isinstance(x2, torch.Tensor):
            x1, x2 = x2, x1  # WLOG x1 is torch.Tensor
        if isinstance(x2, IntervalBoundedTensor):
            z = torch.mul(x2.val, x1)
            lb_mul = torch.mul(x2.lb, x1)
            ub_mul = torch.mul(x2.ub, x1)
            lb_new = torch.min(lb_mul, ub_mul)
            ub_new = torch.max(lb_mul, ub_mul)
            return IntervalBoundedTensor(z, lb_new, ub_new)
        else:
            raise TypeError(x1, x2)
    else:
        if isinstance(x1, IntervalBoundedTensor) and isinstance(x2, IntervalBoundedTensor):
            z = torch.mul(x1.val, x2.val)
            ll = torch.mul(x1.lb, x2.lb)
            lu = torch.mul(x1.lb, x2.ub)
            ul = torch.mul(x1.ub, x2.lb)
            uu = torch.mul(x1.ub, x2.ub)
            stack = torch.stack((ll, lu, ul, uu))
            lb_new = torch.min(stack, dim=0)[0]
            ub_new = torch.max(stack, dim=0)[0]
            return IntervalBoundedTensor(z, lb_new, ub_new)
        else:
            raise TypeError(x1, x2)


def div(x1, x2):
    """Elementwise division of two tensors."""
    if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
        return torch.div(x1, x2)
    if isinstance(x1, IntervalBoundedTensor) and (isinstance(x2, torch.Tensor) or isinstance(x2, int)):
        z = torch.div(x1.val, x2)
        lb_div = torch.div(x1.lb, x2)
        ub_div = torch.div(x1.ub, x2)
        lb_new = torch.min(lb_div, ub_div)
        ub_new = torch.max(lb_div, ub_div)
        return IntervalBoundedTensor(z, lb_new, ub_new)
    elif isinstance(x1, IntervalBoundedTensor) and isinstance(x2, IntervalBoundedTensor):
        assert ((x2.lb > 0) | (x2.ub < 0)).all().item()  # only implement when x2 cannot be zero
        z = torch.div(x1.val, x2.val)
        ll = torch.div(x1.lb, x2.lb)
        lu = torch.div(x1.lb, x2.ub)
        ul = torch.div(x1.ub, x2.lb)
        uu = torch.div(x1.ub, x2.ub)
        stack = torch.stack((ll, lu, ul, uu))
        lb_new = torch.min(stack, dim=0)[0]
        ub_new = torch.max(stack, dim=0)[0]
        return IntervalBoundedTensor(z, lb_new, ub_new)
    else:
        raise TypeError(x1, x2)


def bmm(x1, x2):
    """Batched matrix multiply.

    Args:
      x1: tensor of shape (B, m, p)
      x2: tensor of shape (B, p, n)
    Returns:
      tensor of shape (B, m, n)
    """
    if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
        return torch.matmul(x1, x2)
    elif isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
        swap = False
        if isinstance(x2, torch.Tensor):
            swap = True
            x1, x2 = x2.permute(0, 2, 1), x1.permute(0, 2, 1)  # WLOG x1 is torch.Tensor
        if isinstance(x2, IntervalBoundedTensor):
            z = torch.matmul(x1, x2.val)
            x1_abs = torch.abs(x1)
            mu_cur = (x2.ub + x2.lb) / 2
            r_cur = (x2.ub - x2.lb) / 2
            mu_new = torch.matmul(x1, mu_cur)
            r_new = torch.matmul(x1_abs, r_cur)
            if swap:
                z = z.permute(0, 2, 1)
                mu_new = mu_new.permute(0, 2, 1)
                r_new = r_new.permute(0, 2, 1)
            return IntervalBoundedTensor(z, mu_new - r_new, mu_new + r_new)
        else:
            raise TypeError(x1, x2)
    else:
        if isinstance(x1, IntervalBoundedTensor) and isinstance(x2, IntervalBoundedTensor):
            z = torch.matmul(x1.val, x2.val)
            ll = torch.einsum('ijk,ikl->ijkl', x1.lb, x2.lb)  # B, m, p, n
            lu = torch.einsum('ijk,ikl->ijkl', x1.lb, x2.ub)  # B, m, p, n
            ul = torch.einsum('ijk,ikl->ijkl', x1.ub, x2.lb)  # B, m, p, n
            uu = torch.einsum('ijk,ikl->ijkl', x1.ub, x2.ub)  # B, m, p, n
            stack = torch.stack([ll, lu, ul, uu])
            mins = torch.min(stack, dim=0)[0]  # B, m, p, n
            maxs = torch.max(stack, dim=0)[0]  # B, m, p, n
            lb_new = torch.sum(mins, dim=2)  # B, m, n
            ub_new = torch.sum(maxs, dim=2)  # B, m, n
            return IntervalBoundedTensor(z, lb_new, ub_new)
        else:
            raise TypeError(x1, x2)


def matmul_nneg(x1, x2):
    """Matrix multiply for non-negative matrices (easier than the general case)."""
    if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
        if (x1 < 0).any(): raise ValueError('x1 has negative entries')
        if (x2 < 0).any(): raise ValueError('x2 has negative entries')
        return torch.matmul(x1, x2)
    elif isinstance(x1, torch.Tensor) or isinstance(x2, torch.Tensor):
        swap = False
        if isinstance(x2, torch.Tensor):
            swap = True
            x1, x2 = x2.permute(0, 2, 1), x1.permute(0, 2, 1)  # WLOG x1 is torch.Tensor
        if isinstance(x2, IntervalBoundedTensor):
            if (x1 < 0).any(): raise ValueError('x1 has negative entries')
            if (x2.lb < 0).any(): raise ValueError('x2 has negative lower bounds')
            z = torch.matmul(x1, x2.val)
            lb_new = torch.matmul(x1, x2.lb)
            ub_new = torch.matmul(x1, x2.ub)
            if swap:
                lb_new = lb_new.permute(0, 2, 1)
                ub_new = ub_new.permute(0, 2, 1)
            return IntervalBoundedTensor(z, lb_new, ub_new)
        else:
            raise TypeError(x1, x2)
    else:
        if isinstance(x1, IntervalBoundedTensor) and isinstance(x2, IntervalBoundedTensor):
            if (x1.lb < 0).any(): raise ValueError('x1 has negative lower bounds')
            if (x2.lb < 0).any(): raise ValueError('x2 has negative lower bounds')
            z = torch.matmul(x1.val, x2.val)
            lb_new = torch.matmul(x1.lb, x2.lb)
            ub_new = torch.matmul(x1.ub, x2.ub)
            return IntervalBoundedTensor(z, lb_new, ub_new)
        else:
            raise TypeError(x1, x2)


def cat(tensors, dim=0):
    if all(isinstance(x, torch.Tensor) for x in tensors):
        return torch.cat(tensors, dim=dim)
    tensors_ibp = []
    for x in tensors:
        if isinstance(x, IntervalBoundedTensor):
            tensors_ibp.append(x)
        elif isinstance(x, torch.Tensor):
            tensors_ibp.append(IntervalBoundedTensor(x, x, x))
        else:
            raise TypeError(x)
    return IntervalBoundedTensor(torch.cat([x.val for x in tensors_ibp], dim=dim),
                                 torch.cat([x.lb for x in tensors_ibp], dim=dim),
                                 torch.cat([x.ub for x in tensors_ibp], dim=dim))


def stack(tensors, dim=0):
    if all(isinstance(x, torch.Tensor) for x in tensors):
        return torch.stack(tensors, dim=dim)
    tensors_ibp = []
    for x in tensors:
        if isinstance(x, IntervalBoundedTensor):
            tensors_ibp.append(x)
        elif isinstance(x, torch.Tensor):
            tensors_ibp.append(IntervalBoundedTensor(x, x, x))
        else:
            raise TypeError(x)
    return IntervalBoundedTensor(
        torch.stack([x.val for x in tensors_ibp], dim=dim),
        torch.stack([x.lb for x in tensors_ibp], dim=dim),
        torch.stack([x.ub for x in tensors_ibp], dim=dim))


def pool(func, x, dim):
    """Pooling operations (e.g. mean, min, max).

    For all of these, the pooling passes straight through the bounds.
    """
    if func not in (torch.mean, torch.min, torch.max, torch.sum):
        raise ValueError(func)
    if func in (torch.min, torch.max):
        func_copy = func
        func = lambda *args: func_copy(*args)[0]  # Grab first return value for min/max
    if isinstance(x, torch.Tensor):
        return func(x, dim)
    elif isinstance(x, IntervalBoundedTensor):
        return IntervalBoundedTensor(func(x.val, dim), func(x.lb, dim),
                                     func(x.ub, dim))
    else:
        raise TypeError(x)


def sum(x, *args, **kwargs):
    if isinstance(x, torch.Tensor):
        return torch.sum(x, *args)
    elif isinstance(x, IntervalBoundedTensor):
        return IntervalBoundedTensor(
            torch.sum(x.val, *args, **kwargs),
            torch.sum(x.lb, *args, **kwargs),
            torch.sum(x.ub, *args, **kwargs))
    else:
        raise TypeError(x)


class Activation(nn.Module):
    def __init__(self, func):
        super(Activation, self).__init__()
        self.func = func

    def forward(self, x):
        return activation(self.func, x)


def activation(func, x):
    """Monotonic elementwise activation functions (e.g. ReLU, sigmoid).

    Due to monotonicity, it suffices to evaluate the activation at the endpoints.
    """
    if func not in (F.relu, torch.sigmoid, torch.tanh, torch.exp):
        raise ValueError(func)
    if isinstance(x, torch.Tensor):
        return func(x)
    elif isinstance(x, IntervalBoundedTensor):
        return IntervalBoundedTensor(func(x.val), func(x.lb), func(x.ub))
    else:
        raise TypeError(x)


class LogSoftmax(nn.Module):
    def __init__(self, dim):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return log_softmax(x, self.dim)


def log_softmax(x, dim):
    """logsoftmax operation, requires |dim| to be provided.

    Have to do some weird gymnastics to get vectorization and stability.
    """
    if isinstance(x, torch.Tensor):
        return F.log_softmax(x, dim=dim)
    elif isinstance(x, IntervalBoundedTensor):
        out = F.log_softmax(x.val, dim)
        # Upper-bound on z_i is u_i - log(sum_j(exp(l_j)) + (exp(u_i) - exp(l_i)))
        ub_lb_logsumexp = torch.logsumexp(x.lb, dim, keepdim=True)
        ub_relu = F.relu(x.ub - x.lb)  # ReLU just to prevent cases where lb > ub due to rounding
        # Compute log(exp(u_i) - exp(l_i)) = u_i + log(1 - exp(l_i - u_i)) in 2 different ways
        # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf for further discussion
        # (1) When u_i - l_i <= log(2), use expm1
        ub_log_diff_expm1 = torch.log(-torch.expm1(-ub_relu))
        # (2) When u_i - l_i > log(2), use log1p
        use_log1p = (ub_relu > 0.693)
        ub_relu_log1p = torch.masked_select(ub_relu, use_log1p)
        ub_log_diff_log1p = torch.log1p(-torch.exp(-ub_relu_log1p))
        # NOTE: doing the log1p and then masked_select creates NaN's
        # I think this is likely to be a subtle pytorch bug that unnecessarily
        # propagates NaN gradients.
        ub_log_diff_expm1.masked_scatter_(use_log1p, ub_log_diff_log1p)
        ub_log_diff = x.ub + ub_log_diff_expm1

        ub_scale = torch.max(ub_lb_logsumexp, ub_log_diff)
        ub_log_partition = ub_scale + torch.log(
            torch.exp(ub_lb_logsumexp - ub_scale)
            + torch.exp(ub_log_diff - ub_scale))
        ub_out = x.ub - ub_log_partition

        # Lower-bound on z_i is l_i - log(sum_{j != i}(exp(u_j)) + exp(l_i))
        # Normalizing scores by max_j u_j works except when i = argmax_j u_j, u_i >> argmax_{j != i} u_j, and u_i >> l_i.
        # In this case we normalize by the second value
        lb_ub_max, lb_ub_argmax = torch.max(x.ub, dim, keepdim=True)

        # Make `dim` the last dim for easy argmaxing along it later
        dims = np.append(np.delete(np.arange(len(x.shape)), dim), dim).tolist()
        # Get indices to place `dim` back where it was originally
        rev_dims = np.insert(np.arange(len(x.shape) - 1), dim, len(x.shape) - 1).tolist()
        # Flatten x.ub except for `dim`
        ub_max_masked = x.ub.clone().permute(dims).contiguous().view(-1, x.shape[dim])
        # Get argmax along `dim` and set max indices to -inf
        ub_max_masked[np.arange(np.prod(x.shape) / x.shape[dim]), ub_max_masked.argmax(1)] = -float('inf')
        # Reshape to make it look like x.ub again
        ub_max_masked = ub_max_masked.view(np.array(x.shape).take(dims).tolist()).permute(rev_dims)

        lb_logsumexp_without_argmax = ub_max_masked.logsumexp(dim, keepdim=True)

        lb_ub_exp = torch.exp(x.ub - lb_ub_max)
        lb_cumsum_fwd = torch.cumsum(lb_ub_exp, dim)
        lb_cumsum_bwd = torch.flip(torch.cumsum(torch.flip(lb_ub_exp, [dim]), dim), [dim])
        # Shift the cumulative sums so that i-th element is sum of things before i (after i for bwd)
        pad_fwd = [0] * (2 * len(x.shape))
        pad_fwd[-2 * dim - 2] = 1
        pad_bwd = [0] * (2 * len(x.shape))
        pad_bwd[-2 * dim - 1] = 1
        lb_cumsum_fwd = torch.narrow(F.pad(lb_cumsum_fwd, pad_fwd), dim, 0, x.shape[dim])
        lb_cumsum_bwd = torch.narrow(F.pad(lb_cumsum_bwd, pad_bwd), dim, 1, x.shape[dim])
        lb_logsumexp_without_i = lb_ub_max + torch.log(
            lb_cumsum_fwd + lb_cumsum_bwd)  # logsumexp over everything except i
        lb_logsumexp_without_i.scatter_(dim, lb_ub_argmax, lb_logsumexp_without_argmax)
        lb_scale = torch.max(lb_logsumexp_without_i, x.lb)
        lb_log_partition = lb_scale + torch.log(
            torch.exp(lb_logsumexp_without_i - lb_scale)
            + torch.exp(x.lb - lb_scale))
        lb_out = x.lb - lb_log_partition
        return IntervalBoundedTensor(out, lb_out, ub_out)

    else:
        raise TypeError(x)


def merge(X1, X2):
    if X1 is None:
        return X2
    if X2 is None:
        return X1
    ret = ()
    for (x1, x2) in zip(X1, X2):
        ret += (x1.merge(x2),)
    return ret


def view(X: tuple, *dims):
    ret = ()
    for x in X:
        ret += (x.view(dims),)
    return ret


def where(pred: torch.Tensor, x1: IntervalBoundedTensor, x2: IntervalBoundedTensor):
    return IntervalBoundedTensor(
        torch.where(pred, x1.val, x2.val),
        torch.where(pred, x1.lb, x2.lb),
        torch.where(pred, x1.ub, x2.ub)
    )
