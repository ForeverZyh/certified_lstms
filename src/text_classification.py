"""IBP text classification model."""
import itertools
import glob
import json
import numpy as np
import os
import pickle
import random
import copy
import warnings
import math

from nltk import word_tokenize
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import dgl
import networkx as nx

import attacks
import data_util
import ibp
import vocabulary
from perturbation import Perturbation, Sub, Ins, Del, UNK
from DSL.general_HotFlip import GeneralHotFlipAttack
from DSL.specialized_HotFlip_trees import HotFlipAttackTree
from utils import cons_tree

LOSS_FUNC = nn.BCEWithLogitsLoss()
LOSS_FUNC_KEEP_DIM = nn.BCEWithLogitsLoss(reduction="none")
IMDB_DIR = 'data/aclImdb'
LM_FILE = 'data/lm_scores/imdb_all.txt'
COUNTER_FITTED_FILE = 'data/counter-fitted-vectors.txt'
SST2_DIR = 'data/sst2'


class AdversarialModel(nn.Module):
    def __init__(self):
        super(AdversarialModel, self).__init__()

    def query(self, x, vocab, device, return_bounds=False, attack_surface=None, perturbation=None,
              truncate_to=None):
        """Query the model on a Dataset.

        Args:
          x: a string or a list
          vocab: vocabulary
          device: torch device.

        Returns: list of logits of same length as |examples|.
        """
        if isinstance(x, str):
            dataset = TextClassificationDataset.from_raw_data(
                [(x, 0)], vocab, attack_surface=attack_surface, perturbation=perturbation, truncate_to=truncate_to)
            data = dataset.get_loader(1)
        else:
            dataset = TextClassificationDataset.from_raw_data(
                [(x_, 0) for x_ in x], vocab, attack_surface=attack_surface, perturbation=perturbation,
                truncate_to=truncate_to)
            data = dataset.get_loader(len(x))

        with torch.no_grad():
            batch = data_util.dict_batch_to_device(next(iter(data)), device)
            logits = self.forward(batch, compute_bounds=return_bounds)
            if isinstance(x, str):
                if return_bounds:
                    return logits.val[0].item(), (logits.lb[0].item(), logits.ub[0].item())
                else:
                    return logits[0].item()
            else:
                if return_bounds:
                    return logits.val, (logits.lb, logits.ub)
                else:
                    return logits


def attention_pool(x, mask, layer):
    """Attention pooling

    Args:
      x: batch of inputs, shape (B, n, h)
      mask: binary mask, shape (B, n)
      layer: Linear layer mapping h -> 1
    Returns:
      pooled version of x, shape (B, h)
    """
    attn_raw = layer(x).squeeze(2)  # B, n, 1 -> B, n
    attn_raw = ibp.add(attn_raw, (1 - mask) * -1e20)
    attn_logsoftmax = ibp.log_softmax(attn_raw, 1)
    attn_probs = ibp.activation(torch.exp, attn_logsoftmax)  # B, n
    return ibp.bmm(attn_probs.unsqueeze(1), x).squeeze(1)  # B, 1, n x B, n, h -> B, h


class BOWModel(AdversarialModel):
    """Bag of word vectors + MLP."""

    def __init__(self, word_vec_size, hidden_size, word_mat,
                 pool='max', dropout=0.2, no_wordvec_layer=False):
        super(BOWModel, self).__init__()
        self.pool = pool
        self.no_wordvec_layer = no_wordvec_layer
        self.embs = ibp.Embedding.from_pretrained(word_mat)
        if no_wordvec_layer:
            self.linear_hidden = ibp.Linear(word_vec_size, hidden_size)
        else:
            self.linear_input = ibp.Linear(word_vec_size, hidden_size)
            self.linear_hidden = ibp.Linear(hidden_size, hidden_size)
        self.linear_output = ibp.Linear(hidden_size, 1)
        self.dropout = ibp.Dropout(dropout)
        if self.pool == 'attn':
            self.attn_pool = ibp.Linear(hidden_size, 1)

    def forward(self, batch, compute_bounds=True, cert_eps=1.0):
        """Forward pass of BOWModel.

        Args:
          batch: A batch dict from a TextClassificationDataset with the following keys:
            - x: tensor of word vector indices, size (B, n, 1)
            - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
            - lengths: lengths of sequences, size (B,)
          compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
          cert_eps: Scaling factor for interval bounds of the input
        """
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask']
        lengths = batch['lengths']

        x_vecs = self.embs(x)  # B, n, d
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp.DiscreteChoiceTensorWithUNK):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer:
            z1 = x_vecs
        else:
            z1 = ibp.activation(F.relu, x_vecs)
        z1_masked = z1 * mask.unsqueeze(-1)  # B, n, h
        if self.pool == 'mean':
            z1_pooled = ibp.sum(z1_masked / lengths.to(dtype=torch.float).view(-1, 1, 1), 1)  # B, h
        elif self.pool == 'attn':
            z1_pooled = attention_pool(z1_masked, mask, self.attn_pool)
        else:  # max
            # zero-masking works b/c ReLU guarantees that everything is >= 0
            z1_pooled = ibp.pool(torch.max, z1_masked, 1)  # B, h
        z1_pooled = self.dropout(z1_pooled)
        z2 = ibp.activation(F.relu, self.linear_hidden(z1_pooled))  # B, h
        z2 = self.dropout(z2)
        output = self.linear_output(z2)  # B, 1
        return output


class CNNModel(AdversarialModel):
    """Convolutional neural network.

    Here is the overall architecture:
      1) Rotate word vectors
      2) One convolutional layer
      3) Max/mean pool across all time
      4) Predict with MLP

    """

    def __init__(self, word_vec_size, hidden_size, kernel_size, word_mat,
                 pool='max', dropout=0.2, no_wordvec_layer=False,
                 early_ibp=False, relu_wordvec=True, unfreeze_wordvec=False):
        super(CNNModel, self).__init__()
        cnn_padding = (kernel_size - 1) // 2  # preserves size
        self.pool = pool
        # Ablations
        self.no_wordvec_layer = no_wordvec_layer
        self.early_ibp = early_ibp
        self.relu_wordvec = relu_wordvec
        self.unfreeze_wordvec = False
        # End ablations
        self.embs = ibp.Embedding.from_pretrained(word_mat, freeze=not self.unfreeze_wordvec)
        if no_wordvec_layer:
            self.conv1 = ibp.Conv1d(word_vec_size, hidden_size, kernel_size,
                                    padding=cnn_padding)
        else:
            self.linear_input = ibp.Linear(word_vec_size, hidden_size)
            self.conv1 = ibp.Conv1d(hidden_size, hidden_size, kernel_size,
                                    padding=cnn_padding)
        if self.pool == 'attn':
            self.attn_pool = ibp.Linear(hidden_size, 1)
        self.dropout = ibp.Dropout(dropout)
        self.fc_hidden = ibp.Linear(hidden_size, hidden_size)
        self.fc_output = ibp.Linear(hidden_size, 1)

    def forward(self, batch, compute_bounds=True, cert_eps=1.0):
        """
        Args:
          batch: A batch dict from a TextClassificationDataset with the following keys:
            - x: tensor of word vector indices, size (B, n, 1)
            - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
            - lengths: lengths of sequences, size (B,)
          compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
          cert_eps: Scaling factor for interval bounds of the input
        """
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask']
        lengths = batch['lengths']

        x_vecs = self.embs(x)  # B, n, d
        if self.early_ibp and isinstance(x_vecs, ibp.DiscreteChoiceTensorWithUNK):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp.DiscreteChoiceTensorWithUNK):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer or not self.relu_wordvec:
            z = x_vecs
        else:
            z = ibp.activation(F.relu, x_vecs)  # B, n, h
        z_masked = z * mask.unsqueeze(-1)  # B, n, h
        z_cnn_in = z_masked.permute(0, 2, 1)  # B, h, n
        c1 = ibp.activation(F.relu, self.conv1(z_cnn_in))  # B, h, n
        c1_masked = c1 * mask.unsqueeze(1)  # B, h, n
        if self.pool == 'mean':
            fc_in = ibp.sum(c1_masked / lengths.to(dtype=torch.float).view(-1, 1, 1), 2)  # B, h
        elif self.pool == 'attn':
            fc_in = attention_pool(c1_masked.permute(0, 2, 1), mask, self.attn_pool)  # B, h
        else:
            # zero-masking works b/c ReLU guarantees that everything is >= 0
            fc_in = ibp.pool(torch.max, c1_masked, 2)  # B, h
        fc_in = self.dropout(fc_in)
        fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, 1
        return output


class LSTMModel(AdversarialModel):
    """LSTM text classification model.

    Here is the overall architecture:
      1) Rotate word vectors
      2) Feed to bi-LSTM
      3) Max/mean pool across all time
      4) Predict with MLP

    """

    def __init__(self, word_vec_size, hidden_size, word_mat, device, pool='max', dropout=0.2,
                 no_wordvec_layer=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.pool = pool
        self.no_wordvec_layer = no_wordvec_layer
        self.device = device
        self.embs = ibp.Embedding.from_pretrained(word_mat)
        if no_wordvec_layer:
            self.lstm = ibp.LSTM(word_vec_size, hidden_size, bidirectional=True)
        else:
            self.linear_input = ibp.Linear(word_vec_size, hidden_size)
            self.lstm = ibp.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.dropout = ibp.Dropout(dropout)
        self.fc_hidden = ibp.Linear(2 * hidden_size, hidden_size)
        self.fc_output = ibp.Linear(hidden_size, 1)

    def forward(self, batch, compute_bounds=True, cert_eps=1.0, analysis_mode=False):
        """
        Args:
          batch: A batch dict from a TextClassificationDataset with the following keys:
            - x: tensor of word vector indices, size (B, n, 1)
            - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
            - lengths: lengths of sequences, size (B,)
          compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
          cert_eps: Scaling factor for interval bounds of the input
        """
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask']
        lengths = batch['lengths']

        B = x.shape[0]
        x_vecs = self.embs(x)  # B, n, d
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp.DiscreteChoiceTensorWithUNK):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer:
            z = x_vecs
        else:
            z = ibp.activation(F.relu, x_vecs)  # B, n, h
        h0 = torch.zeros((B, 2 * self.hidden_size), device=self.device)  # B, 2*h
        c0 = torch.zeros((B, 2 * self.hidden_size), device=self.device)  # B, 2*h
        if analysis_mode:
            h_mat, c_mat, lstm_analysis = self.lstm(z, (h0, c0), mask=mask, analysis_mode=True)  # B, n, 2*h each
        else:
            h_mat, c_mat = self.lstm(z, (h0, c0), mask=mask)  # B, n, 2*h each
        h_masked = h_mat * mask.unsqueeze(2)
        if self.pool == 'mean':
            fc_in = ibp.sum(h_masked / lengths.to(dtype=torch.float).view(-1, 1, 1), 1)  # B, 2*h
        else:
            raise NotImplementedError()
        fc_in = self.dropout(fc_in)
        fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, 1
        if analysis_mode:
            return output, h_mat, c_mat, lstm_analysis
        return output


class LSTMDPModel(AdversarialModel):
    """LSTM text classification model.
    A specific implementation for Sub, Ins, Del

    Here is the overall architecture:
      1) Rotate word vectors
      2) Feed to bi-LSTM
      3) Max/mean pool across all time
      4) Predict with MLP

    """

    def __init__(self, word_vec_size, hidden_size, word_mat, device, pool='max', dropout=0.2,
                 no_wordvec_layer=False, bidirectional=True, perturbation=None, baseline=False):
        super(LSTMDPModel, self).__init__()
        assert perturbation is not None
        self.perturbation = perturbation
        self.adv_attack = False  # use to compute the gradients
        self.out_x_vecs = None  # use to compute the gradients
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.word_vec_size = word_vec_size
        self.pool = pool
        if pool != 'final' and not baseline:
            warnings.warn("pool = %s is not implemented! for non length-preserving perturbations" % pool,
                          RuntimeWarning)
        self.no_wordvec_layer = no_wordvec_layer
        self.device = device
        self.embs = ibp.Embedding.from_pretrained(word_mat)
        if no_wordvec_layer:
            self.lstm = ibp.LSTMDP(word_vec_size, hidden_size, perturbation, bidirectional=bidirectional,
                                   baseline=baseline)
        else:
            self.linear_input = ibp.Linear(word_vec_size, hidden_size)
            self.lstm = ibp.LSTMDP(hidden_size, hidden_size, perturbation, bidirectional=bidirectional,
                                   baseline=baseline)
        self.dropout = ibp.Dropout(dropout)
        if bidirectional:
            self.fc_hidden = ibp.Linear(hidden_size * 2, hidden_size)
        else:
            self.fc_hidden = ibp.Linear(hidden_size, hidden_size)
        self.fc_output = ibp.Linear(hidden_size, 1)

    def get_embed(self, x, ret_len=None):
        """
        :param x: tensor of word vector indices
        :param ret_len: the return length, if None, the length is equal to len(x). if len(x) < ret_len, we add padding
        :return: np array with shape (ret_len, word_vec_size)
        """
        if ret_len is None:
            ret_len = len(x)
        ret = np.zeros((ret_len, self.word_vec_size))
        with torch.no_grad():
            ret[:len(x)] = self.embs(x.unsqueeze(0))[0].cpu().numpy()
        return ret

    def get_grad(self, x, y):
        """
        :param x: tensor of word vector indices (B, len)
        :param y: tensor of labels (B, 1)
        :return: np array with shape (B, len, word_vec_size)
        """
        self.adv_attack = True

        logits = self.forward(x, compute_bounds=False)
        loss = LOSS_FUNC(logits, y)
        loss.backward()

        self.adv_attack = False

        return self.out_x_vecs.grad.cpu().numpy()

    def forward(self, batch, compute_bounds=True, cert_eps=1.0, analysis_mode=False):
        """
        Args:
          batch: A batch dict from a TextClassificationDataset with the following keys:
            - x: tensor of word vector indices, size (B, n, 1)
            - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
            - lengths: lengths of sequences, size (B,)
          compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
          cert_eps: Scaling factor for interval bounds of the input
        """
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask']
        lengths = batch['lengths']

        B = x.shape[0]
        x_vecs = self.embs(x)  # B, n, d
        if self.adv_attack:
            x_vecs.requires_grad = True
            self.out_x_vecs = x_vecs
        z_interval = None
        unk_mask = None
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
            if isinstance(x_vecs, ibp.DiscreteChoiceTensorWithUNK):
                x_interval = x_vecs.to_interval_bounded(eps=cert_eps)
                unk_mask = x_vecs.unk_mask.to(self.device)
                x_vecs = x_vecs.val
                z_interval = ibp.activation(F.relu, x_interval)  # B, n, h

            z = ibp.activation(F.relu, x_vecs)  # B, n, h
        else:
            if isinstance(x_vecs, ibp.DiscreteChoiceTensorWithUNK):
                x_interval = x_vecs.to_interval_bounded(eps=cert_eps)
                x_vecs = x_vecs.val
                z_interval = x_interval

            z = x_vecs

        h0 = torch.zeros((B, self.hidden_size * (2 if self.bidirectional else 1)), device=self.device)  # B, h
        c0 = torch.zeros((B, self.hidden_size * (2 if self.bidirectional else 1)), device=self.device)  # B, h
        if analysis_mode:
            h_mat, c_mat, lstm_analysis = self.lstm(z, z_interval, (h0, c0), mask=mask,
                                                    analysis_mode=True, unk_mask=unk_mask)  # B, n, h each
        else:
            h_mat, c_mat = self.lstm(z, z_interval, (h0, c0), mask=mask, unk_mask=unk_mask)  # B, n, h each

        if self.pool == 'mean':
            h_masked = h_mat * mask.unsqueeze(2)  # only need to mask the state sequence
            fc_in = ibp.sum(h_masked / lengths.to(dtype=torch.float).view(-1, 1, 1), 1)  # B, h
        elif self.pool == 'final':
            # don't need to mask the final state
            fc_in = h_mat[:, -1, :]
        else:
            raise NotImplementedError()
        fc_in = self.dropout(fc_in)
        fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, 1
        if analysis_mode:
            return output, h_mat, c_mat, lstm_analysis
        return output


class LSTMDPModelASCC(LSTMDPModel):
    def __init__(self, word_vec_size, hidden_size, word_mat, device, filepath, pool='max', dropout=0.2,
                 perturbation=None, baseline=False):
        super(LSTMDPModelASCC, self).__init__(word_vec_size, hidden_size, word_mat, device, pool,
                                              dropout, False, True, perturbation, baseline)
        filename = os.path.join(filepath, "bilstm_adv_best.pth")
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        state_dict = checkpoint['net']

        def switch_last_two_hidden(x):
            # from W_ii|W_if|W_ig|W_io to W_ii|W_if|W_io|W_ig
            return torch.cat([x[:2*hidden_size], x[3*hidden_size:], x[2*hidden_size:3*hidden_size]], dim=0)

        self.lstm.i2h.weight.data = switch_last_two_hidden(state_dict["bilstm.weight_ih_l0"])
        self.lstm.i2h.bias.data = switch_last_two_hidden(state_dict["bilstm.bias_ih_l0"])
        self.lstm.h2h.weight.data = switch_last_two_hidden(state_dict["bilstm.weight_hh_l0"])
        self.lstm.h2h.bias.data = switch_last_two_hidden(state_dict["bilstm.bias_hh_l0"])
        self.lstm.back_i2h.weight.data = switch_last_two_hidden(state_dict["bilstm.weight_ih_l0_reverse"])
        self.lstm.back_i2h.bias.data = switch_last_two_hidden(state_dict["bilstm.bias_ih_l0_reverse"])
        self.lstm.back_h2h.weight.data = switch_last_two_hidden(state_dict["bilstm.weight_hh_l0_reverse"])
        self.lstm.back_h2h.bias.data = switch_last_two_hidden(state_dict["bilstm.bias_hh_l0_reverse"])
        self.linear_input.weight.data = state_dict["linear_transform_embd_1.weight"]
        self.linear_input.bias.data = state_dict["linear_transform_embd_1.bias"]
        self.fc_hidden.weight.data = state_dict["hidden1.weight"]
        self.fc_hidden.bias.data = state_dict["hidden1.bias"]
        fc_output_w = state_dict["hidden2label.weight"]
        fc_output_b = state_dict["hidden2label.bias"]
        self.fc_output.weight.data = (fc_output_w[1] - fc_output_w[0]).unsqueeze(
            0)  # from (2, hidden_dim) to (1, hidden_dim)
        self.fc_output.bias.data = (fc_output_b[1] - fc_output_b[0]).unsqueeze(0)


class LSTMDPGeneralModel(AdversarialModel):
    """LSTM text classification model.
    A general implementation for any transformations

    Here is the overall architecture:
      1) Rotate word vectors
      2) Feed to bi-LSTM
      3) Max/mean pool across all time
      4) Predict with MLP

    """

    def __init__(self, word_vec_size, hidden_size, word_mat, device, pool='max', dropout=0.2,
                 no_wordvec_layer=False, perturbation=None):
        super(LSTMDPGeneralModel, self).__init__()
        assert perturbation is not None
        self.adv_attack = False  # use to compute the gradients
        self.out_x_vecs = None  # use to compute the gradients
        self.hidden_size = hidden_size
        self.word_vec_size = word_vec_size
        self.pool = pool
        self.no_wordvec_layer = no_wordvec_layer
        self.device = device
        self.embs = ibp.Embedding.from_pretrained(word_mat)
        if no_wordvec_layer:
            self.lstm = ibp.LSTMDPGeneral(word_vec_size, hidden_size, Perturbation.dumy_perturbation(perturbation),
                                          device)
        else:
            self.linear_input = ibp.Linear(word_vec_size, hidden_size)
            self.lstm = ibp.LSTMDPGeneral(hidden_size, hidden_size, Perturbation.dumy_perturbation(perturbation),
                                          device)
        self.dropout = ibp.Dropout(dropout)
        self.fc_output = ibp.Linear(hidden_size, 1)
        self.fc_hidden = ibp.Linear(hidden_size, hidden_size)

    def get_embed(self, x, ret_len=None):
        """
        :param x: tensor of word vector indices
        :param ret_len: the return length, if None, the length is equal to len(x). if len(x) < ret_len, we add padding
        :return: np array with shape (ret_len, word_vec_size)
        """
        if ret_len is None:
            ret_len = len(x)
        ret = np.zeros((ret_len, self.word_vec_size))
        with torch.no_grad():
            ret[:len(x)] = self.embs(x.unsqueeze(0))[0].cpu().numpy()
        return ret

    def get_grad(self, x, y):
        """
        :param x: tensor of word vector indices (B, len)
        :param y: tensor of labels (B, 1)
        :return: np array with shape (B, len, word_vec_size)
        """
        self.adv_attack = True

        logits = self.forward(x, compute_bounds=False)
        loss = LOSS_FUNC(logits, y)
        loss.backward()

        self.adv_attack = False

        return self.out_x_vecs.grad.cpu().numpy()

    def query(self, x, vocab, device, return_bounds=False, attack_surface=None, perturbation=None):
        """Query the model on a Dataset.

        Args:
          x: a string or a list
          vocab: vocabulary
          device: torch device.

        Returns: list of logits of same length as |examples|.
        """
        if perturbation is None:
            return super(LSTMDPGeneralModel, self).query(x, vocab, device, return_bounds, attack_surface, perturbation)
        if isinstance(x, str):
            dataset = TextClassificationDatasetGeneral.from_raw_data(
                [(x, 0)], vocab, attack_surface=attack_surface, perturbation=perturbation)
            data = dataset.get_loader(1)
        else:
            dataset = TextClassificationDatasetGeneral.from_raw_data(
                [(x_, 0) for x_ in x], vocab, attack_surface=attack_surface, perturbation=perturbation)
            data = dataset.get_loader(len(x))

        with torch.no_grad():
            batch = data_util.dict_batch_to_device(next(iter(data)), device)
            logits = self.forward(batch, compute_bounds=return_bounds)
            if isinstance(x, str):
                if return_bounds:
                    return logits.val[0].item(), (logits.lb[0].item(), logits.ub[0].item())
                else:
                    return logits[0].item()
            else:
                if return_bounds:
                    return logits.val, (logits.lb, logits.ub)
                else:
                    return logits

    def forward(self, batch, compute_bounds=True, cert_eps=1.0):
        """
        Args:
          batch: A batch dict from a TextClassificationDataset with the following keys:
            - x: tensor of word vector indices, size (B, n, 1)
            - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
            - lengths: lengths of sequences, size (B,)
          compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
          cert_eps: Scaling factor for interval bounds of the input
        """
        x = batch['x']
        mask = batch['mask']
        lengths = batch['lengths']

        B = x.shape[0]
        if not compute_bounds and isinstance(x, ibp.DiscreteChoiceTensorWithUNK):
            x = x.val
        x_vecs = self.embs(x)  # B, n, d
        if self.adv_attack:
            x_vecs.requires_grad = True
            self.out_x_vecs = x_vecs

        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
            z = ibp.activation(F.relu, x_vecs)  # B, n, h
        else:
            z = x_vecs

        # compute z_interval
        trans_output = None
        if compute_bounds:
            trans_o, trans_phi = batch['trans_output']
            z_interval = []
            for o in trans_o:
                if isinstance(o, ibp.DiscreteChoiceTensorTrans):
                    o = self.embs(o)
                    if not self.no_wordvec_layer:
                        o = self.linear_input(o).to_interval_bounded(eps=cert_eps)
                        z_interval.append(ibp.activation(F.relu, o))
                    else:
                        z_interval.append(o.to_interval_bounded(eps=cert_eps))
                else:
                    z_interval.append(o)
            trans_output = (z_interval, trans_phi)

        h0 = torch.zeros((B, self.hidden_size), device=self.device)  # B, h
        c0 = torch.zeros((B, self.hidden_size), device=self.device)  # B, h
        h_mat, c_mat, lengths_interval = self.lstm(z, trans_output, (h0, c0), lengths, mask=mask)  # B, n, h each

        max_length = h_mat.shape[1]
        if self.pool == 'mean':
            if trans_output is None:
                h_masked = h_mat * mask.unsqueeze(2)  # only need to mask the state sequence
                fc_in = ibp.sum(h_masked / lengths.to(dtype=torch.float).view(-1, 1, 1), 1)  # B, h
            else:
                fc_in = [None] * B
                for i in range(B):
                    assert lengths_interval.lb[i].item() <= lengths[i].item() <= lengths_interval.ub[i].item()
                    cum_sum = ibp.sum(h_mat[i, :lengths_interval.lb[i].item()], 0)
                    fc_in[i] = cum_sum / lengths_interval.lb[i].item()
                    for j in range(lengths_interval.lb[i].item(), lengths_interval.ub[i].item()):
                        cum_sum += h_mat[i, j]
                        if j + 1 == lengths[i].item():
                            fc_in[i] = (cum_sum / (j + 1)).merge(fc_in[i])
                        else:
                            fc_in[i] = fc_in[i].merge(cum_sum / (j + 1))
                fc_in = ibp.stack(fc_in, dim=0)
        elif self.pool == 'final':
            if trans_output is None:
                fc_in = h_mat[:, -1, :]  # B, h
            else:
                fc_in = [None] * B
                for i in range(B):
                    assert lengths_interval.lb[i].item() <= lengths[i].item() <= lengths_interval.ub[i].item()
                    for j in range(lengths_interval.lb[i].item() - 1, lengths_interval.ub[i].item()):
                        if fc_in[i] is None:
                            fc_in[i] = h_mat[i, j]
                        else:
                            if j + 1 == lengths[i].item():
                                fc_in[i] = h_mat[i, j].merge(fc_in[i])
                            else:
                                fc_in[i] = fc_in[i].merge(h_mat[i, j])
                fc_in = ibp.stack(fc_in, dim=0)
        else:
            raise NotImplementedError()
        fc_in = self.dropout(fc_in)
        fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, 1
        return output


class LSTMFinalStateModel(AdversarialModel):
    """LSTM text classification model that uses final hidden state."""

    def __init__(self, word_vec_size, hidden_size, word_mat, device, dropout=0.2,
                 no_wordvec_layer=False):
        super(LSTMFinalStateModel, self).__init__()
        self.hidden_size = hidden_size
        self.no_wordvec_layer = no_wordvec_layer
        self.device = device
        self.embs = ibp.Embedding.from_pretrained(word_mat)
        if no_wordvec_layer:
            self.lstm = ibp.LSTM(word_vec_size, hidden_size)
        else:
            self.linear_input = ibp.Linear(word_vec_size, hidden_size)
            self.lstm = ibp.LSTM(hidden_size, hidden_size)
        self.dropout = ibp.Dropout(dropout)
        self.fc_hidden = ibp.Linear(hidden_size, hidden_size)
        self.fc_output = ibp.Linear(hidden_size, 1)

    def forward(self, batch, compute_bounds=True, cert_eps=1.0, analysis_mode=False):
        """
        Args:
          batch: A batch dict from a TextClassificationDataset with the following keys:
            - x: tensor of word vector indices, size (B, n, 1)
            - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
            - lengths: lengths of sequences, size (B,)
          compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
          cert_eps: Scaling factor for interval bounds of the input
        """
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask']
        lengths = batch['lengths']

        B = x.shape[0]
        x_vecs = self.embs(x)  # B, n, d
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp.DiscreteChoiceTensorWithUNK):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer:
            z = x_vecs
        else:
            z = ibp.activation(F.relu, x_vecs)  # B, n, h
        h0 = torch.zeros((B, self.hidden_size), device=self.device)  # B, h
        c0 = torch.zeros((B, self.hidden_size), device=self.device)  # B, h
        if analysis_mode:
            h_mat, c_mat, lstm_analysis = self.lstm(z, (h0, c0), mask=mask, analysis_mode=True)  # B, n, h each
        else:
            h_mat, c_mat = self.lstm(z, (h0, c0), mask=mask)  # B, n, h each
        h_final = h_mat[:, -1, :]  # B, h
        fc_in = self.dropout(h_final)
        fc_hidden = ibp.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, 1
        if analysis_mode:
            return output, h_mat, c_mat, lstm_analysis
        return output


class Adversary(object):
    """An Adversary tries to fool a model on a given example."""

    def __init__(self, attack_surface):
        self.attack_surface = attack_surface

    def run(self, model, dataset, device, opts=None):
        """Run adversary on a dataset.

        Args:
          model: a TextClassificationModel.
          dataset: a TextClassificationDataset.
          device: torch device.
        Returns: pair of
          - list of 0-1 adversarial loss of same length as |dataset|
          - list of list of adversarial examples (each is just a text string)
        """
        raise NotImplementedError


class RSAdversary(Adversary):
    """An Adversary that uses RS to test the random smoothing accuracy.
    """

    def __init__(self, attack_surface, perturbation):
        super(RSAdversary, self).__init__(attack_surface)
        self.deltas = Perturbation.str2deltas(perturbation)

    def run(self, model, dataset, device, opts=None):
        is_correct = []
        delta = np.sqrt(math.log(2 / opts.RS_MC_error) / 2 / opts.RS_sample_num)
        for x, y in tqdm(dataset.raw_data):
            words = [x.lower() for x in x.split()]
            choices = self.attack_surface.get_swaps(words)

            correct = 0
            total_size, f = RSAdversary.cal_total_size(self.deltas[2], words, choices)
            if total_size <= opts.RS_sample_num:
                iter = ExhaustiveAdversary.DelDupSubWord(*self.deltas, words, choices, batch_size=50)
                total_sample = total_size
            else:
                iter = RSAdversary.RandomSubWord(self.deltas[2], words, choices, f, sample_num=opts.RS_sample_num)
                total_sample = opts.RS_sample_num
            tem_tv = self.attack_surface.get_tv(words)
            remaining = total_sample

            for batch_x in iter:
                all_raw = [' '.join(x_new) for x_new in batch_x]
                preds = model.query(all_raw, dataset.vocab, device)
                correct += len([p for p in preds if p * (2 * y - 1) > 0])
                remaining -= len(all_raw)
                # early stopping to speed up (about 2X)
                if total_sample == total_size:
                    if correct > 0.5 * total_sample:
                        break
                    if correct + remaining < 0.5 * total_sample:
                        break
                else:
                    if correct / total_sample * 1.0 - 1. + np.prod(tem_tv[:self.deltas[2]]) > 0.5 + delta:
                        break
                    if (correct + remaining) / total_sample * 1.0 - 1. + np.prod(tem_tv[:self.deltas[2]]) < 0.5 + delta:
                        break

            if total_sample == total_size:
                is_correct.append(correct > 0.5 * total_sample)
            else:
                is_correct.append(
                    correct / total_sample * 1.0 - 1. + np.prod(tem_tv[:self.deltas[2]]) > 0.5 + delta)
            print(is_correct[-1], ",\tcert%=", correct / total_sample * 1.0, ",\ttotal sample=", total_sample,
                  ",\tearly stopping at", correct, "/", total_sample - remaining)

        return is_correct, []

    @staticmethod
    def cal_total_size(c, x, choices):
        end_pos = len(x)
        f = np.zeros((end_pos + 1, c + 1), dtype=np.int64)
        f[0, 0] = 1
        for i in range(1, end_pos + 1):
            f[i, 0] = f[i - 1, 0]
            for j in range(1, c + 1):
                f[i, j] = f[i - 1, j] + f[i - 1, j - 1] * len(choices[i - 1])

        total_size = np.sum(f[-1])
        return total_size, f

    @staticmethod
    def RandomSubWord(c, x, choices, f, sample_num=5000, batch_size=50):
        end_pos = len(x)
        total_size = np.sum(f[-1])
        X = []
        for _ in range(0, sample_num, batch_size):
            rand_numbers = np.random.randint(0, total_size, batch_size)
            for rand_number in rand_numbers:
                x1 = copy.copy(x)
                j = 0
                while f[-1, j] <= rand_number:
                    rand_number -= f[-1, j]
                    j += 1
                for i in range(end_pos, 0, -1):
                    if j == 0:
                        break
                    if rand_number >= f[i - 1, j]:
                        rand_number -= f[i - 1, j]
                        x1[i - 1] = choices[i - 1][rand_number // f[i - 1, j - 1]]
                        rand_number %= f[i - 1, j - 1]
                        j -= 1

                X.append(x1)
            yield X
            X = []


class HotFlipAdversary(Adversary):
    """An Adversary that uses HotFlip to search for worst case perturbations.
    """

    def __init__(self, victim_model, perturbation, tree_attack=False):
        super(HotFlipAdversary, self).__init__(None)
        self.victim_model = victim_model
        from DSL.transformation import Ins, Del, Sub, Trans1, Trans2, Trans3, Trans4
        if tree_attack:
            self.attack = HotFlipAttackTree(eval(perturbation))
        else:
            self.attack = GeneralHotFlipAttack(eval(perturbation))

    def run(self, model, dataset, device, opts=None):
        is_correct = []
        adv_exs = []
        for x, y in tqdm(dataset.raw_data):
            # First query the example itself
            orig_pred = model.query(x, dataset.vocab, device)
            if orig_pred * (2 * y - 1) <= 0:
                print('ORIGINAL PREDICTION WAS WRONG')
                is_correct.append(0)
                adv_exs.append(x)
                continue

            words = [x.lower() for x in x.split()]
            words = [w for w in words if w in dataset.vocab]  # test words can be outside the vocabulary, we omit them
            # chances are that the substituted words are outside the vocabulary due to the pos tags are different from
            # train and dev set. If that happens, the substituted words will be seen as UNK
            ans = self.attack.gen_adv(self.victim_model, words, y, opts.adv_beam, self.victim_model.get_embed)
            is_correct_single = True
            all_raw = [' '.join(x_new) for x_new in ans]
            preds = model.query(all_raw, dataset.vocab, device)

            cur_adv_exs = [all_raw[i] for i, p in enumerate(preds)
                           if p * (2 * y - 1) <= 0]
            if len(cur_adv_exs) > 0:
                print(cur_adv_exs)
                adv_exs.append(cur_adv_exs)
                is_correct_single = False

            print('HotFlipAdversary: "%s" -> %d options' % (x, is_correct_single))
            is_correct.append(is_correct_single)
            if is_correct_single:
                adv_exs.append([])

        return is_correct, adv_exs

    def run_tree(self, model, dataset, device, trainset_vocab, PAD_WORD, opts=None):
        is_correct = []
        adv_exs = []
        tree_data_vocab_key_list = list(trainset_vocab.keys())
        for example in tqdm(dataset.examples):
            # First query the example itself
            g = example["trees"][0]
            x = " ".join(example["rawx"])
            root_ids = [i for i in range(g.number_of_nodes()) if g.out_degrees(i) == 0]
            target = g.ndata['y'].long()[root_ids].item()
            orig_pred = model.query(example["trees"], dataset.vocab, device, trainset_vocab, PAD_WORD,
                                    return_bounds=False)
            orig_pred = orig_pred[0]

            def is_logits_correct(x):
                return all(x[i].item() < x[target].item() for i in range(x.shape[0]) if i != target)

            if not is_logits_correct(orig_pred):
                print('ORIGINAL PREDICTION WAS WRONG')
                is_correct.append(0)
                adv_exs.append(x)
                continue

            text = example["rawx"]

            is_correct_single = True
            ans = self.attack.gen_adv(self.victim_model, g, text, opts.adv_beam, trainset_vocab)
            all_raw = [
                " ".join([tree_data_vocab_key_list[x.item()] for x in g.ndata["x"] if x.item() != PAD_WORD])
                for g in ans]
            preds = model.query(ans, dataset.vocab, device, trainset_vocab, PAD_WORD)
            cur_adv_exs = [all_raw[i] for i, p in enumerate(preds) if not is_logits_correct(p)]
            if len(cur_adv_exs) > 0:
                print(cur_adv_exs)
                adv_exs.append(cur_adv_exs)
                is_correct_single = False

            print('HotFlipAdversary: "%s" -> %d options' % (x, is_correct_single))
            is_correct.append(is_correct_single)
            if is_correct_single:
                adv_exs.append([])

        return is_correct, adv_exs


class EvalAdversary(Adversary):
    """Evaluating the size of the perturbation space
    """

    def __init__(self, attack_surface, perturbation):
        super(EvalAdversary, self).__init__(attack_surface)
        self.perturbation = perturbation
        self.deltas = Perturbation.str2deltas(perturbation)

    def run(self, model, dataset, device, opts=None):
        size = []
        ngram_start = 2
        ngram_end = 10
        ngrams = [{} for _ in range(ngram_end)]
        for x, y in tqdm(dataset.raw_data):
            words = [x.lower() for x in x.split()]
            if '!' in words or "?" in words:
                print("here")
            for i in range(len(words)):
                for j in range(ngram_start, ngram_end):
                    if i + j <= len(words):
                        keys = tuple(words[i: i + j])
                        ngrams[j][keys] = ngrams[j].setdefault(keys, 0) + 1
            swaps = self.attack_surface.get_swaps(words)
            choices = [[s for s in cur_swaps if s in dataset.vocab] for w, cur_swaps in zip(words, swaps)]

            end_pos = len(words)
            f = np.zeros((end_pos + 1, self.deltas[0] + 1, self.deltas[1] + 1, self.deltas[2] + 1), dtype=np.int64)
            f[0, 0, 0, 0] = 1
            for i in range(1, end_pos + 1):
                for cnt_del in range(self.deltas[0] + 1):
                    for cnt_ins in range(self.deltas[1] + 1):
                        for cnt_sub in range(self.deltas[2] + 1):
                            f[i, cnt_del, cnt_ins, cnt_sub] = f[i - 1, cnt_del, cnt_ins, cnt_sub]
                            if cnt_del > 0 and words[i - 1] in {"a", "and", "the", "of", "to"}:
                                f[i, cnt_del, cnt_ins, cnt_sub] += f[i - 1, cnt_del - 1, cnt_ins, cnt_sub]
                            if cnt_ins > 0:
                                f[i, cnt_del, cnt_ins, cnt_sub] += f[i - 1, cnt_del, cnt_ins - 1, cnt_sub]
                            if cnt_sub > 0 and len(choices[i - 1]) > 0:
                                f[i, cnt_del, cnt_ins, cnt_sub] += f[i - 1, cnt_del, cnt_ins, cnt_sub - 1] * len(
                                    choices[i - 1])

            size.append(np.sum(f[-1]))

        print("Average: ", np.mean(size), "\tMax: ", np.max(size), "\tMin: ", np.min(size))
        for j in range(ngram_start, ngram_end):
            ngram_list = list(ngrams[j].items())
            ngram_list.sort(key=lambda x: -x[1])
            ngrams[j] = dict(ngram_list[:50])
        np.save("./EvalSize", ngrams)
        return [0], []


class ExhaustiveAdversary(Adversary):
    """An Adversary that exhaustively tries all allowed perturbations.

    Only practical for short sentences.
    """

    def __init__(self, attack_surface, perturbation):
        super(ExhaustiveAdversary, self).__init__(attack_surface)
        self.perturbation = perturbation
        self.deltas = Perturbation.str2deltas(perturbation)

    def run(self, model, dataset, device, opts=None):
        is_correct = []
        adv_exs = []
        for x, y in tqdm(dataset.raw_data):
            # First query the example itself
            orig_pred, (orig_lb, orig_ub) = model.query(
                x, dataset.vocab, device, return_bounds=True,
                attack_surface=self.attack_surface, perturbation=self.perturbation, truncate_to=opts.truncate_to)
            orig_pred = model.query(x, dataset.vocab, device, return_bounds=False)
            assert orig_lb - 1e-5 <= orig_pred <= orig_ub + 1e-5
            cert_correct = (orig_lb * (2 * y - 1) > 0) and (orig_ub * (2 * y - 1) > 0)
            print('Logit bounds: %.6f <= %.6f <= %.6f, cert_correct=%s' % (
                orig_lb, orig_pred, orig_ub, cert_correct))
            if orig_pred * (2 * y - 1) <= 0:
                print('ORIGINAL PREDICTION WAS WRONG')
                is_correct.append(0)
                adv_exs.append(x)
                continue
            elif cert_correct:
                print('CERTIFY CORRECT')
                is_correct.append(1)
                adv_exs.append([])
                continue

            words = [x.lower() for x in x.split()]
            swaps = self.attack_surface.get_swaps(words)
            choices = [[s for s in cur_swaps if s in dataset.vocab] for w, cur_swaps in zip(words, swaps)]

            is_correct_single = True
            for batch_x in ExhaustiveAdversary.DelDupSubWord(*self.deltas, words, choices, batch_size=10):
                all_raw = [' '.join(x_new) for x_new in batch_x]
                preds = model.query(all_raw, dataset.vocab, device, truncate_to=opts.truncate_to)
                if not (orig_lb - 1e-5 <= preds.min() and orig_ub + 1e-5 >= preds.max()):
                    print("Fail! ", preds.min(), preds.max())
                    print("Min: ", all_raw[int(preds.min(dim=0)[1][0])])
                    print("Max: ", all_raw[int(preds.max(dim=0)[1][0])])
                    _ = input("Press any key to continue...")
                cur_adv_exs = [all_raw[i] for i, p in enumerate(preds)
                               if p * (2 * y - 1) <= 0]
                if len(cur_adv_exs) > 0:
                    print(cur_adv_exs)
                    adv_exs.append(cur_adv_exs)
                    is_correct_single = False
                    break

            # TODO: count the number
            print('ExhaustiveAdversary: "%s" -> %d options' % (x, is_correct_single))
            is_correct.append(is_correct_single)
            if is_correct_single:
                adv_exs.append([])

        return is_correct, adv_exs

    def run_tree(self, model, dataset, device, trainset_vocab, PAD_WORD, opts=None):
        is_correct = []
        adv_exs = []
        tree_data_vocab_key_list = list(trainset_vocab.keys())
        for example in tqdm(dataset.examples):
            # First query the example itself
            g = example["trees"][0]
            x = " ".join(example["rawx"])
            root_ids = [i for i in range(g.number_of_nodes()) if g.out_degrees(i) == 0]
            target = g.ndata['y'].long()[root_ids].item()
            orig_pred, interval_pred = model.query(
                example["trees"], dataset.vocab, device, trainset_vocab, PAD_WORD, return_bounds=True,
                attack_surface=self.attack_surface, perturbation=self.perturbation)
            orig_pred = orig_pred[0]
            interval_pred = interval_pred[0]

            def is_logits_correct(x):
                return all(x[i].item() < x[target].item() for i in range(x.shape[0]) if i != target)

            cert_correct = is_logits_correct(interval_pred)
            print('Orig logits: %s, Cert bounds: %s, target: %d, cert_correct=%s' % (
                ",".join(["%.4f" % x.item() for x in orig_pred]),
                ",".join(["%.4f" % x.item() for x in interval_pred]),
                target, cert_correct))
            if not is_logits_correct(orig_pred):
                print('ORIGINAL PREDICTION WAS WRONG')
                is_correct.append(0)
                adv_exs.append(x)
                continue
            elif cert_correct:
                print('CERTIFY CORRECT')
                is_correct.append(1)
                adv_exs.append([])
                continue

            text = example["rawx"]
            swaps = self.attack_surface.get_swaps(text)
            choices = [[s for s in cur_swaps if s in trainset_vocab] for w, cur_swaps in zip(text, swaps) if
                       w in trainset_vocab]

            is_correct_single = True
            for batch_x in ExhaustiveAdversary.DelDupSubTree(*self.deltas, text, g, trainset_vocab, choices,
                                                             batch_size=opts.batch_size):
                all_raw = [
                    " ".join([tree_data_vocab_key_list[x.item()] for x in g.ndata["x"] if x.item() != PAD_WORD])
                    for g in batch_x]
                preds = model.query(batch_x, dataset.vocab, device, trainset_vocab, PAD_WORD)
                preds_logits = preds.max(dim=0)[0]
                preds_min = preds.min(dim=0)[0]
                preds_logits[target] = preds_min[target]
                if not (preds_min[target] + 1e-5 >= interval_pred[target] and all(
                        preds_logits[i] <= interval_pred[i] + 1e-5 for i in range(interval_pred.shape[0]) if
                        i != target)):
                    print("Fail! ", preds_logits, interval_pred, target)
                    print(all_raw)
                    _ = input("Press any key to continue...")
                cur_adv_exs = [all_raw[i] for i, p in enumerate(preds) if not is_logits_correct(p)]
                if len(cur_adv_exs) > 0:
                    print(cur_adv_exs)
                    adv_exs.append(cur_adv_exs)
                    is_correct_single = False
                    break

            # TODO: count the number
            print('ExhaustiveAdversary: "%s" -> %d options' % (x, is_correct_single))
            is_correct.append(is_correct_single)
            if is_correct_single:
                adv_exs.append([])

        return is_correct, adv_exs

    @staticmethod
    def DelDupSubWord(a, b, c, x, choices, batch_size=64, del_set={"a", "and", "the", "of", "to"}):
        end_pos = len(x)

        valid_sub_poss = [i for i in range(end_pos) if len(choices[i]) > 0]
        X = []
        for sub in range(c, -1, -1):
            for sub_poss in itertools.combinations(tuple(valid_sub_poss), sub):
                sub_pos_strs = []
                for sub_pos in sub_poss:
                    sub_pos_strs.append(choices[sub_pos])
                for sub_pos_str in itertools.product(*sub_pos_strs):
                    x3 = copy.copy(x)
                    for i, sub_pos in enumerate(sub_poss):
                        x3[sub_pos] = sub_pos_str[i]
                    valid_dup_poss = [i for i in range(end_pos) if i not in sub_poss]
                    for dup in range(b, -1, -1):
                        for dup_poss in itertools.combinations(tuple(valid_dup_poss), dup):
                            valid_del_poss = [i for i in range(end_pos) if
                                              (i not in dup_poss) and (i not in sub_poss) and x[i] in del_set]
                            for delete in range(a, -1, -1):
                                for del_poss in itertools.combinations(tuple(valid_del_poss), delete):
                                    x2 = []
                                    copy_point = 0
                                    while copy_point < end_pos:
                                        if copy_point in dup_poss:
                                            x2.append(x3[copy_point])
                                            x2.append(x3[copy_point])
                                            copy_point += 1
                                        elif copy_point in del_poss:
                                            copy_point += 1
                                        else:
                                            x2.append(x3[copy_point])
                                            copy_point += 1

                                    X.append(x2)
                                    if len(X) == batch_size:
                                        yield X
                                        X = []

        if len(X) > 0:
            yield X

    @staticmethod
    def DelDupSubTree(a, b, c, x, tree, vocab, choices, batch_size=64, del_set={"a", "and", "the", "of", "to"}):
        end_pos = len(x)

        valid_sub_poss = [i for i in range(end_pos) if len(choices[i]) > 0]
        X = []
        # list of trees that are constructed by (word_id, phi, f)
        # word_id is in x
        # phi [1 -> Del, 2 -> Dup, 3 -> Sub]
        # f [None or syn]
        for sub in range(c, -1, -1):
            for sub_poss in itertools.combinations(tuple(valid_sub_poss), sub):
                sub_pos_strs = []
                for sub_pos in sub_poss:
                    sub_pos_strs.append(choices[sub_pos])
                for sub_pos_str in itertools.product(*sub_pos_strs):
                    f = [None] * len(x)
                    for i, sub_pos in enumerate(sub_poss):
                        f[sub_pos] = sub_pos_str[i]
                    valid_dup_poss = [i for i in range(end_pos) if i not in sub_poss]
                    for dup in range(b, -1, -1):
                        for dup_poss in itertools.combinations(tuple(valid_dup_poss), dup):
                            valid_del_poss = [i for i in range(end_pos) if
                                              (i not in dup_poss) and (i not in sub_poss) and x[i] in del_set]

                            for delete in range(a, -1, -1):
                                for del_poss in itertools.combinations(tuple(valid_del_poss), delete):
                                    phi = [0] * len(x)
                                    for sub_pos in sub_poss:
                                        phi[sub_pos] = 3
                                    for dup_pos in dup_poss:
                                        phi[dup_pos] = 2
                                    for del_pos in del_poss:
                                        phi[del_pos] = 1

                                    X.append(cons_tree(x, phi, f, tree, vocab))
                                    if len(X) == batch_size:
                                        yield X
                                        X = []

        if len(X) > 0:
            yield X


class GeneralExhaustiveAdversary(Adversary):
    """An Adversary that exhaustively tries all allowed perturbations.

    Only practical for short sentences.
    """
    vocab = None

    def __init__(self, attack_surface, perturbation):
        super(GeneralExhaustiveAdversary, self).__init__(attack_surface)
        self.perturbation = perturbation

    def run(self, model, dataset, device, opts=None):
        is_correct = []
        adv_exs = []
        sizes = []
        GeneralExhaustiveAdversary.vocab = dataset.vocab
        for x, y in tqdm(dataset.raw_data):
            # First query the example itself
            orig_pred_, (orig_lb, orig_ub) = model.query(
                x, dataset.vocab, device, return_bounds=True,
                attack_surface=self.attack_surface, perturbation=self.perturbation)
            orig_pred = model.query(x, dataset.vocab, device, return_bounds=False)
            assert abs(orig_pred_ - orig_pred) < 1e-5
            assert orig_lb - 1e-5 <= orig_pred <= orig_ub + 1e-5
            cert_correct = (orig_lb * (2 * y - 1) > 0) and (orig_ub * (2 * y - 1) > 0)
            print('Logit bounds: %.6f <= %.6f <= %.6f, cert_correct=%s' % (
                orig_lb, orig_pred, orig_ub, cert_correct))
            if orig_pred * (2 * y - 1) <= 0:
                print('ORIGINAL PREDICTION WAS WRONG')
                is_correct.append(0)
                adv_exs.append(x)
                continue
            elif cert_correct:
                print('CERTIFY CORRECT')
                is_correct.append(1)
                adv_exs.append([])
                continue

            words = [x.lower() for x in x.split()]

            is_correct_single = True
            perturb = Perturbation(self.perturbation, words, dataset.vocab, attack_surface=self.attack_surface)
            words = perturb.ipt
            cnt_eval = 0
            for batch_x in GeneralExhaustiveAdversary.gen_batch(words, perturb, batch_size=10):
                all_raw = [' '.join(x_new) for x_new in batch_x]
                cnt_eval += len(all_raw)
                preds = model.query(all_raw, dataset.vocab, device)
                if not (orig_lb - 1e-5 <= preds.min() and orig_ub + 1e-5 >= preds.max()):
                    print("Fail! ", preds.min(), preds.max())
                    print("Min: ", all_raw[int(preds.min(dim=0)[1][0])])
                    print("Max: ", all_raw[int(preds.max(dim=0)[1][0])])
                    _ = input("Press any key to continue...")
                cur_adv_exs = [all_raw[i] for i, p in enumerate(preds)
                               if p * (2 * y - 1) <= 0]
                if len(cur_adv_exs) > 0:
                    print(cur_adv_exs)
                    adv_exs.append(cur_adv_exs)
                    is_correct_single = False
                    break

            # TODO: count the number
            if is_correct_single:
                print("ExhaustiveAdversary: %s -> all correct, %d samples evaluated." % (x, cnt_eval))
            else:
                print("ExhaustiveAdversary: %s -> found wrong example, %d samples evaluated." % (x, cnt_eval))
            sizes.append(cnt_eval)
            is_correct.append(is_correct_single)
            if is_correct_single:
                adv_exs.append([])

        print("Max: %d, Mean: %.2f" % (max(sizes), sum(sizes) * 1.0 / len(sizes)))
        return is_correct, adv_exs

    @staticmethod
    def gen_batch(ipt, perturb, batch_size=10):
        trans_o = []
        for tran in perturb.trans:
            tran_o = tran.gen_output_for_dp()
            ret_o = []
            for i, o in enumerate(tran_o):
                if len(o) > 0:
                    ret_o.append((i, o))

            trans_o.append(ret_o)

        X = []
        for x in GeneralExhaustiveAdversary.gen_all(ipt, 0, list(zip(perturb.trans, trans_o)), [], np.zeros(len(ipt))):
            X.append(x)
            if len(X) == batch_size:
                yield X
                X = []

        if len(X) > 0:
            yield X

    @staticmethod
    def gen_all(ipt, step, perturb, trace, covered):
        """
        generate all perturbations
        :param ipt: the original input
        :param step: the current transformation
        :param perturb: zip of the perturbation spaces and the transform outputs
        :param trace: a list of (start_pos, end_pos, tran_res)
        :param covered: a 0-1 array, 0 means not covered yet. To be used checking the non-overlapping
        :return:
        """
        if step == len(perturb):
            # the end of DFS
            trace.sort(key=lambda x: x[:2])
            new_x = []
            pre = 0
            for (start_pos, end_pos, tran_res) in trace:
                new_x += ipt[pre:start_pos]
                new_x += list(tran_res)
                pre = end_pos
            new_x += ipt[pre:]

            yield new_x
            return

        tran, tran_o = perturb[step]
        for delta in range(tran.delta, -1, -1):
            for o_s in itertools.combinations(tran_o, delta):
                # check for non-overlapping
                if any(np.any(covered[o[0]: tran.s + o[0]]) for o in o_s):
                    continue
                post_covered = np.copy(covered)
                for o in o_s:
                    post_covered[o[0]: tran.s + o[0]] = 1

                tran_res_choices = [o[1] for o in o_s]
                for tran_reses in itertools.product(*tran_res_choices):
                    if any(any(w not in GeneralExhaustiveAdversary.vocab for w in tran_res) for tran_res in tran_reses):
                        continue

                    appended_trace = [(o[0], tran.s + o[0], tran_res) for o, tran_res in zip(o_s, tran_reses)]
                    for x in GeneralExhaustiveAdversary.gen_all(ipt, step + 1, perturb, trace + appended_trace,
                                                                post_covered):
                        yield x


class GreedyAdversary(Adversary):
    """An adversary that picks a random word and greedily tries perturbations."""

    def __init__(self, attack_surface, num_epochs=10, num_tries=2, margin_goal=0.0):
        super(GreedyAdversary, self).__init__(attack_surface)
        self.num_epochs = num_epochs
        self.num_tries = num_tries
        self.margin_goal = margin_goal

    def run(self, model, dataset, device, opts=None):
        is_correct = []
        adv_exs = []
        for x, y in tqdm(dataset.raw_data):
            # First query the example itself
            orig_pred, bounds = model.query(TextClassificationDataset.from_raw_data(
                [(x, y)], dataset.vocab, self.attack_surface), device, return_bounds=True)
            orig_pred, (orig_lb, orig_ub) = orig_pred[0], bounds[0]
            cert_correct = (orig_lb * (2 * y - 1) > 0) and (orig_ub * (2 * y - 1) > 0)
            print('Logit bounds: %.6f <= %.6f <= %.6f, cert_correct=%s' % (
                orig_lb, orig_pred, orig_ub, cert_correct))
            if orig_pred * (2 * y - 1) <= 0:
                print('ORIGINAL PREDICTION WAS WRONG')
                is_correct.append(0)
                adv_exs.append(x)
                continue
            # Now run adversarial search
            words = [x.lower() for x in x.split()]
            swaps = self.attack_surface.get_swaps(words)
            choices = [[w] + cur_swaps for w, cur_swaps in zip(words, swaps)]
            found = False
            for try_idx in range(self.num_tries):
                cur_words = list(words)
                for epoch in range(self.num_epochs):
                    word_idxs = list(range(len(choices)))
                    random.shuffle(word_idxs)
                    for i in word_idxs:
                        cur_raw = []
                        for w_new in choices[i]:
                            cur_raw.append((' '.join(cur_words[:i] + [w_new] + cur_words[i + 1:]), y))
                        cur_dataset = TextClassificationDataset.from_raw_data(cur_raw, dataset.vocab)
                        preds = model.query(cur_dataset, device)
                        margins = [p * (2 * y - 1) for p in preds]
                        best_idx = min(enumerate(margins), key=lambda x: x[1])[0]
                        cur_words[i] = choices[i][best_idx]
                        if margins[best_idx] < self.margin_goal:
                            found = True
                            is_correct.append(0)
                            adv_exs.append([' '.join(cur_words)])
                            print('ADVERSARY SUCCESS on ("%s", %d): Found "%s" with margin %.2f' % (
                                x, y, adv_exs[-1], margins[best_idx]))
                            if cert_correct:
                                print('^^ CERT CORRECT THOUGH')
                            break
                    if found: break
                if found: break
            else:
                is_correct.append(1)
                adv_exs.append([])
                print('ADVERSARY FAILURE on ("%s", %d)' % (x, y))
        return is_correct, adv_exs


class GeneticAdversary(Adversary):
    """An adversary that runs a genetic attack."""

    def __init__(self, attack_surface, num_iters=20, pop_size=60, margin_goal=0.0):
        super(GeneticAdversary, self).__init__(attack_surface)
        self.num_iters = num_iters
        self.pop_size = pop_size
        self.margin_goal = margin_goal

    def perturb(self, words, choices, model, y, vocab, device):
        if all(len(c) == 1 for c in choices): return words
        good_idxs = [i for i, c in enumerate(choices) if len(c) > 1]
        idx = random.sample(good_idxs, 1)[0]
        x_list = [' '.join(words[:idx] + [w_new] + words[idx + 1:])
                  for w_new in choices[idx]]
        preds = [model.query(x, vocab, device) for x in x_list]
        margins = [p * (2 * y - 1) for p in preds]
        best_idx = min(enumerate(margins), key=lambda x: x[1])[0]
        cur_words = list(words)
        cur_words[idx] = choices[idx][best_idx]
        return cur_words, margins[best_idx]

    def run(self, model, dataset, device, opts=None):
        is_correct = []
        adv_exs = []
        for x, y in tqdm(dataset.raw_data):
            # First query the example itself
            orig_pred, (orig_lb, orig_ub) = model.query(
                x, dataset.vocab, device, return_bounds=True,
                attack_surface=self.attack_surface)
            cert_correct = (orig_lb * (2 * y - 1) > 0) and (orig_ub * (2 * y - 1) > 0)
            print('Logit bounds: %.6f <= %.6f <= %.6f, cert_correct=%s' % (
                orig_lb, orig_pred, orig_ub, cert_correct))
            if orig_pred * (2 * y - 1) <= 0:
                print('ORIGINAL PREDICTION WAS WRONG')
                is_correct.append(0)
                adv_exs.append(x)
                continue
            # Now run adversarial search
            words = [x.lower() for x in x.split()]
            swaps = self.attack_surface.get_swaps(words)
            choices = [[w] + cur_swaps for w, cur_swaps in zip(words, swaps)]
            found = False
            population = [self.perturb(words, choices, model, y, dataset.vocab, device)
                          for i in range(self.pop_size)]
            for g in range(self.num_iters):
                best_idx = min(enumerate(population), key=lambda x: x[1][1])[0]
                print('Iteration %d: %.6f' % (g, population[best_idx][1]))
                if population[best_idx][1] < self.margin_goal:
                    found = True
                    is_correct.append(0)
                    adv_exs.append(' '.join(population[best_idx][0]))
                    print('ADVERSARY SUCCESS on ("%s", %d): Found "%s" with margin %.2f' % (
                        x, y, adv_exs[-1], population[best_idx][1]))
                    if cert_correct:
                        print('^^ CERT CORRECT THOUGH')
                    break
                new_population = [population[best_idx]]
                margins = np.array([m for c, m in population])
                adv_probs = 1 / (1 + np.exp(margins)) + 1e-6
                # Sigmoid of negative margin, for probabilty of wrong class
                # Add 1e-6 for numerical stability
                sample_probs = adv_probs / np.sum(adv_probs)
                for i in range(1, self.pop_size):
                    parent1 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
                    parent2 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
                    child = [random.sample([w1, w2], 1)[0] for (w1, w2) in zip(parent1, parent2)]
                    child_mut, new_margin = self.perturb(child, choices, model, y,
                                                         dataset.vocab, device)
                    new_population.append((child_mut, new_margin))
                population = new_population
            else:
                is_correct.append(1)
                adv_exs.append([])
                print('ADVERSARY FAILURE on ("%s", %d)' % (x, y))
        return is_correct, adv_exs


def load_datasets(device, opts):
    """
    Loads text classification datasets given opts on the device and returns the dataset.
    If a data cache is specified in opts and the cached data there is of the same class
      as the one specified in opts, uses the cache. Otherwise reads from the raw dataset
      files specified in OPTS.
    Returns:
      - train_data:  EntailmentDataset - Processed training dataset
      - dev_data: Optional[EntailmentDataset] - Processed dev dataset if raw dev data was found or
          dev_frac was specified in opts
      - word_mat: torch.Tensor
      - attack_surface: AttackSurface - defines the adversarial attack surface
    """
    if opts.use_toy_data:
        data_class = ToyClassificationDataset
    elif opts.dataset == "Imdb":
        data_class = IMDBDataset
    elif opts.dataset == "SST2":
        data_class = SST2Dataset
    else:
        raise NotImplementedError
    try:
        with open(os.path.join(opts.data_cache_dir, 'train_data.pkl'), 'rb') as infile:
            train_data = pickle.load(infile)
            if not isinstance(train_data, data_class) and not (
                    opts.model == "lstm-dp-general" and isinstance(train_data, TextClassificationDatasetGeneral)):
                raise Exception("Cached dataset of wrong class: {}".format(type(train_data)))
        with open(os.path.join(opts.data_cache_dir, 'dev_data.pkl'), 'rb') as infile:
            dev_data = pickle.load(infile)
            if not isinstance(dev_data, data_class) and not (
                    opts.model == "lstm-dp-general" and isinstance(dev_data, TextClassificationDatasetGeneral)):
                raise Exception("Cached dataset of wrong class: {}".format(type(train_data)))
        with open(os.path.join(opts.data_cache_dir, 'word_mat.pkl'), 'rb') as infile:
            word_mat = pickle.load(infile)
        with open(os.path.join(opts.data_cache_dir, 'attack_surface.pkl'), 'rb') as infile:
            attack_surface = pickle.load(infile)
        print("Loaded data from {}.".format(opts.data_cache_dir))
    except Exception:
        if opts.use_toy_data:
            attack_surface = ToyClassificationAttackSurface(ToyClassificationDataset.VOCAB_LIST)
        elif opts.use_lm:
            attack_surface = attacks.LMConstrainedAttackSurface.from_files(
                opts.neighbor_file, opts.imdb_lm_file)
        elif opts.use_a3t_settings:
            attack_surface = attacks.A3TWordSubstitutionAttackSurface.from_file(opts.pddb_file, opts.use_fewer_sub)
        elif opts.use_RS_settings:
            attack_surface = attacks.RSAttackSurface.from_files()
        elif opts.use_none_settings:
            attack_surface = attacks.NoneAttackSurface.from_file()
        else:
            attack_surface = attacks.WordSubstitutionAttackSurface.from_file(opts.neighbor_file)
        print('Reading dataset.')
        if opts.dataset == "Imdb":
            raw_data = data_class.get_raw_data(opts.imdb_dir, test=opts.test)
        elif opts.dataset == "SST2":
            raw_data = data_class.get_raw_data(opts.sst2_dir, test=opts.test)
        else:
            raise NotImplementedError
        if opts.model == "lstm-dp-ascc":
            # load word_mat
            filename = os.path.join(opts.out_dir, "bilstm_adv_best.pth")
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
            state_dict = checkpoint['net']
            word_mat = state_dict["embedding.weight"]
            # load vocab
            vocab = vocabulary.Vocabulary()
            index_word = dict(np.load(os.path.join(opts.out_dir, "index_word.npy"), allow_pickle=True).item())
            vocab.word_list += [None] * len(index_word)
            for i in range(1, len(index_word) + 1):
                vocab.word_list[i] = index_word[i]
            word_index = dict(np.load(os.path.join(opts.out_dir, "word_index.npy"), allow_pickle=True).item())
            vocab.word2index.update(word_index)
        else:
            word_set = raw_data.get_word_set(attack_surface,
                                             use_counter_vocab=not opts.use_a3t_settings and not opts.use_none_settings)
            vocab, word_mat = vocabulary.Vocabulary.read_word_vecs(word_set, opts.glove_dir, opts.glove, device)
        if opts.model == "lstm-dp-general":
            train_data = TextClassificationDatasetGeneral.from_raw_data(raw_data.train_data, vocab, attack_surface,
                                                                        downsample_to=opts.downsample_to,
                                                                        downsample_shard=opts.downsample_shard,
                                                                        truncate_to=opts.truncate_to,
                                                                        perturbation=opts.perturbation)
            dev_data = TextClassificationDatasetGeneral.from_raw_data(raw_data.dev_data, vocab, attack_surface,
                                                                      downsample_to=opts.downsample_to,
                                                                      downsample_shard=opts.downsample_shard,
                                                                      truncate_to=opts.truncate_to,
                                                                      perturbation=opts.perturbation)
        else:
            train_data = data_class.from_raw_data(raw_data.train_data, vocab, attack_surface,
                                                  downsample_to=opts.downsample_to,
                                                  downsample_shard=opts.downsample_shard,
                                                  truncate_to=opts.truncate_to,
                                                  perturbation=opts.perturbation)
            dev_data = data_class.from_raw_data(raw_data.dev_data, vocab, attack_surface,
                                                downsample_to=opts.downsample_to,
                                                downsample_shard=opts.downsample_shard,
                                                truncate_to=opts.truncate_to,
                                                perturbation=opts.perturbation)
        if opts.data_cache_dir:
            with open(os.path.join(opts.data_cache_dir, 'train_data.pkl'), 'wb') as outfile:
                pickle.dump(train_data, outfile)
            with open(os.path.join(opts.data_cache_dir, 'dev_data.pkl'), 'wb') as outfile:
                pickle.dump(dev_data, outfile)
            with open(os.path.join(opts.data_cache_dir, 'word_mat.pkl'), 'wb') as outfile:
                pickle.dump(word_mat, outfile)
            with open(os.path.join(opts.data_cache_dir, 'attack_surface.pkl'), 'wb') as outfile:
                pickle.dump(attack_surface, outfile)
    return train_data, dev_data, word_mat, attack_surface


def num_correct_multi_classes(model_output, gold_labels, num_classes):
    """
    Given the output of model and gold labels returns number of correct and certified correct
    predictions
    Args:
      - model_output: output of the model, could be ibp.IntervalBoundedTensor or torch.Tensor
      - gold_labels: torch.Tensor, should be of size 1 per sample, 1 for positive 0 for negative
    Returns:
      - num_correct: int - number of correct predictions from the actual model output
      - num_cert_correct - number of bounds-certified correct predictions if the model_output was an
          IntervalBoundedTensor, 0 otherwise.
    """
    if isinstance(model_output, ibp.IntervalBoundedTensor):
        logits = model_output.val
        num_cert_correct = 0
        for i, y in enumerate(gold_labels):
            y = int(y)
            num_cert_correct += all(
                j == y or (model_output.lb[i][y].item() > model_output.ub[i][j].item()) for j in range(num_classes))
    else:
        logits = model_output
        num_cert_correct = 0
    pred = torch.argmax(logits, dim=1)
    num_correct = sum(
        pred[i].item() == y.item() for i, y in enumerate(gold_labels)
    )
    return num_correct, num_cert_correct


def num_correct(model_output, gold_labels):
    """
    Given the output of model and gold labels returns number of correct and certified correct
    predictions
    Args:
      - model_output: output of the model, could be ibp.IntervalBoundedTensor or torch.Tensor
      - gold_labels: torch.Tensor, should be of size 1 per sample, 1 for positive 0 for negative
    Returns:
      - num_correct: int - number of correct predictions from the actual model output
      - num_cert_correct - number of bounds-certified correct predictions if the model_output was an
          IntervalBoundedTensor, 0 otherwise.
    """
    if isinstance(model_output, ibp.IntervalBoundedTensor):
        logits = model_output.val
        num_cert_correct = sum(
            all((b * (2 * y - 1)).item() > 0 for b in (model_output.lb[i], model_output.ub[i]))
            for i, y in enumerate(gold_labels)
        )
    else:
        logits = model_output
        num_cert_correct = 0
    num_correct = sum(
        (logits[i] * (2 * y - 1)).item() > 0 for i, y in enumerate(gold_labels)
    )
    return num_correct, num_cert_correct


def load_model(word_mat, device, opts):
    """
    Try to load a model on the device given the word_mat and opts.
    Tries to load a model from the given or latest checkpoint if specified in the opts.
    Otherwise instantiates a new model on the device.
    """
    if opts.model == 'bow':
        model = BOWModel(
            vocabulary.GLOVE_CONFIGS[opts.glove]['size'], opts.hidden_size, word_mat,
            pool=opts.pool, dropout=opts.dropout_prob, no_wordvec_layer=opts.no_wordvec_layer).to(device)
    elif opts.model == 'cnn':
        model = CNNModel(
            vocabulary.GLOVE_CONFIGS[opts.glove]['size'], opts.hidden_size, opts.kernel_size,
            word_mat, pool=opts.pool, dropout=opts.dropout_prob, no_wordvec_layer=opts.no_wordvec_layer,
            early_ibp=opts.early_ibp, relu_wordvec=not opts.no_relu_wordvec, unfreeze_wordvec=opts.unfreeze_wordvec).to(
            device)
    elif opts.model == 'lstm':
        model = LSTMModel(
            vocabulary.GLOVE_CONFIGS[opts.glove]['size'], opts.hidden_size,
            word_mat, device, pool=opts.pool, dropout=opts.dropout_prob, no_wordvec_layer=opts.no_wordvec_layer).to(
            device)
    elif opts.model == "lstm-dp":
        model = LSTMDPModel(
            vocabulary.GLOVE_CONFIGS[opts.glove]['size'], opts.hidden_size,
            word_mat, device, pool=opts.pool, dropout=opts.dropout_prob, no_wordvec_layer=opts.no_wordvec_layer,
            perturbation=opts.perturbation, bidirectional=not opts.no_bidirectional, baseline=opts.baseline).to(device)
    elif opts.model == "lstm-dp-ascc":
        model = LSTMDPModelASCC(
            100001, opts.hidden_size, word_mat, device, opts.out_dir, pool=opts.pool, dropout=opts.dropout_prob,
            perturbation=opts.perturbation, baseline=opts.baseline).to(device)
    elif opts.model == "lstm-dp-general":
        model = LSTMDPGeneralModel(
            vocabulary.GLOVE_CONFIGS[opts.glove]['size'], opts.hidden_size,
            word_mat, device, pool=opts.pool, dropout=opts.dropout_prob, no_wordvec_layer=opts.no_wordvec_layer,
            perturbation=opts.perturbation).to(device)
    elif opts.model == 'lstm-final-state':
        model = LSTMFinalStateModel(
            vocabulary.GLOVE_CONFIGS[opts.glove]['size'], opts.hidden_size,
            word_mat, device, dropout=opts.dropout_prob, no_wordvec_layer=opts.no_wordvec_layer).to(device)
    if opts.load_dir:
        try:
            if opts.load_ckpt is None:
                load_fn = sorted(glob.glob(os.path.join(opts.load_dir, 'model-checkpoint-[0-9]+.pth')))[-1]
            else:
                load_fn = os.path.join(opts.load_dir, 'model-checkpoint-%d.pth' % opts.load_ckpt)
            print('Loading model from %s.' % load_fn)
            state_dict = dict(torch.load(load_fn, map_location=torch.device('cpu')))
            state_dict['embs.weight'] = model.embs.weight
            model.load_state_dict(state_dict)
            print('Finished loading model.')
        except Exception as ex:
            print("Couldn't load model, starting anew: {}".format(ex))
    return model


class RawClassificationDataset(data_util.RawDataset):
    """
    Dataset that only holds x,y as (str, str) tuples
    """

    def get_word_set(self, attack_surface, use_counter_vocab=True):
        with open(COUNTER_FITTED_FILE) as f:
            counter_vocab = set([line.split(' ')[0] for line in f])
        word_set = set()
        for x, y in self.data:
            words = [w.lower() for w in x.split(' ')]
            for w in words:
                word_set.add(w)
            try:
                swaps = attack_surface.get_swaps(words)
                for cur_swaps in swaps:
                    for w in cur_swaps:
                        word_set.add(w)
            except KeyError:
                # For now, ignore things not in attack surface
                # If we really need them, later code will throw an error
                pass
        if use_counter_vocab:
            return word_set & counter_vocab
        else:
            return word_set


class TextClassificationDataset(data_util.ProcessedDataset):
    """
    Dataset that holds processed example dicts
    """

    @classmethod
    def from_raw_data(cls, raw_data, vocab, attack_surface=None, truncate_to=None,
                      downsample_to=None, downsample_shard=0, perturbation=None):
        if downsample_to:
            raw_data = raw_data[downsample_shard * downsample_to:(downsample_shard + 1) * downsample_to]
        examples = []
        if perturbation is not None:
            deltas = Perturbation.str2deltas(perturbation)
        for x, y in raw_data:
            all_words = [w.lower() for w in x.split()]
            ins_delta = 0
            if perturbation is not None:
                perturb = Perturbation(perturbation, all_words, vocab, attack_surface=attack_surface)
                ins_delta = deltas[perturb.Ins_idx]
                choices = perturb.get_output_for_baseline_final_state()
                choices = [[x for x in choice if x == UNK or x in vocab] for choice in choices]
                words = perturb.ipt
            elif attack_surface is not None:
                raise AttributeError
            else:
                words = [w for w in all_words if w in vocab]  # Delete UNK words

            # truncate and add padding
            if truncate_to:
                words = words[:truncate_to]
                if perturbation is not None:
                    choices = choices[:truncate_to]
                for _ in range(truncate_to - len(words)):
                    words.append(vocabulary.UNK_TOKEN)
                    if perturbation is not None:
                        choices.append([vocabulary.UNK_TOKEN])

            if len(words) == 0:
                continue
            word_idxs = [vocab.get_index(w) for w in words] + [0] * ins_delta  # append dummy words
            x_torch = torch.tensor(word_idxs).view(1, -1, 1)  # (1, T, d)
            if perturbation is not None:
                choices = choices + [[] for _ in range(ins_delta)]  # append dummy choices
                unk_mask = torch.tensor([0 if UNK in c_list else 1 for c_list in choices],
                                        dtype=torch.long).unsqueeze(0)  # (1, T)
                choices_word_idxs = [
                    torch.tensor([vocab.get_index(c) for c in c_list if c != UNK], dtype=torch.long) for c_list in
                    choices
                ]
                # if any(0 in c.view(-1).tolist() for c in choices_word_idxs):
                #     raise ValueError("UNK tokens found")
                choices_torch = pad_sequence(choices_word_idxs, batch_first=True).unsqueeze(2).unsqueeze(
                    0)  # (1, T, C, 1)
                choices_mask = (choices_torch.squeeze(-1) != 0).long()  # (1, T, C)
            elif attack_surface is not None:
                raise AttributeError
            else:
                choices_torch = x_torch.view(1, -1, 1, 1)  # (1, T, 1, 1)
                choices_mask = torch.ones_like(x_torch.view(1, -1, 1))
                unk_mask = None

            mask_torch = torch.ones((1, len(word_idxs)))
            for i in range(1, ins_delta + 1):
                mask_torch[0][-i] = 0
            if unk_mask is None:
                unk_mask = torch.ones((1, len(word_idxs)))
            x_bounded = ibp.DiscreteChoiceTensorWithUNK(x_torch, choices_torch, choices_mask, mask_torch, unk_mask)
            y_torch = torch.tensor(y, dtype=torch.float).view(1, 1)
            lengths_torch = torch.tensor(len(word_idxs)).view(1)
            examples.append(dict(x=x_bounded, y=y_torch, mask=mask_torch, lengths=lengths_torch))
        return cls(raw_data, vocab, examples)

    @staticmethod
    def example_len(example):
        return example['x'].shape[1]

    @staticmethod
    def collate_examples(examples):
        """
        Turns a list of examples into a workable batch:
        """
        if len(examples) == 1:
            return examples[0]
        B = len(examples)
        max_len = max(ex['x'].shape[1] for ex in examples)
        x_vals = []
        choice_mats = []
        choice_masks = []
        y = torch.zeros((B, 1))
        lengths = torch.zeros((B,), dtype=torch.long)
        masks = torch.zeros((B, max_len))
        unk_masks = torch.zeros((B, max_len))
        for i, ex in enumerate(examples):
            x_vals.append(ex['x'].val)
            choice_mats.append(ex['x'].choice_mat)
            choice_masks.append(ex['x'].choice_mask)
            cur_len = ex['x'].shape[1]
            masks[i, :cur_len] = ex['x'].sequence_mask[0]
            unk_masks[i, :cur_len] = ex['x'].unk_mask[0] if ex['x'].unk_mask is not None else 1
            y[i, 0] = ex['y']
            lengths[i] = ex['lengths'][0]
        x_vals = data_util.multi_dim_padded_cat(x_vals, 0).long()
        choice_mats = data_util.multi_dim_padded_cat(choice_mats, 0).long()
        choice_masks = data_util.multi_dim_padded_cat(choice_masks, 0).long()
        return {'x': ibp.DiscreteChoiceTensorWithUNK(x_vals, choice_mats, choice_masks, masks, unk_masks),
                'y': y, 'mask': masks, 'lengths': lengths}


class TextClassificationDatasetGeneral(data_util.ProcessedDataset):
    """
    Dataset that holds processed example dicts
    """

    @classmethod
    def from_raw_data(cls, raw_data, vocab, attack_surface=None, truncate_to=None,
                      downsample_to=None, downsample_shard=0, perturbation=None):
        assert perturbation is not None
        if downsample_to:
            raw_data = raw_data[downsample_shard * downsample_to:(downsample_shard + 1) * downsample_to]
        examples = []
        for x, y in raw_data:
            all_words = [w.lower() for w in x.split()]
            trans_o_id = []
            trans_phi = []
            perturb = Perturbation(perturbation, all_words, vocab, attack_surface=attack_surface)
            words = perturb.ipt
            for tran in perturb.trans:
                choices = tran.gen_output_for_dp()
                phi = []
                o = [[[] for _ in range(tran.t)] for _ in choices]
                for start_pos in range(len(choices)):
                    phi.append(False)
                    for choice in choices[start_pos]:
                        # We give up this choice if any of the words are out of vocab
                        # This can vary between different implementations
                        if all(choice[j] in vocab for j in range(tran.t)):
                            phi[-1] = True  # there is a valid choice
                            for j in range(tran.t):
                                o[start_pos][j].append(vocab.get_index(choice[j]))

                trans_phi.append(torch.tensor(phi, dtype=torch.bool).unsqueeze(0))
                trans_o_id.append(o)
                # trans_o_id contains a list of o. o has length len(x) and the shape (tran.t, #choice).

            if truncate_to:
                words = words[:truncate_to]
            if len(words) == 0:
                continue
            word_idxs = [vocab.get_index(w) for w in words]
            x_torch = torch.tensor(word_idxs).view(1, -1, 1)  # (1, T, 1)
            trans_o = []
            mask_torch = torch.ones((1, len(word_idxs)))
            for tran_id, tran in enumerate(perturb.trans):
                choices_torch = []
                choices_mask = []
                for start_pos in range(len(words)):
                    # choices_torch will be a list of tensors with shape (1, tran.t, C, 1)
                    choices_torch.append(
                        torch.tensor(trans_o_id[tran_id][start_pos], dtype=torch.long).unsqueeze(0).unsqueeze(-1))
                    choices_mask.append((choices_torch[-1].squeeze(-1) != 0).long())  # (1, tran.t, C)
                choices_torch = data_util.multi_dim_padded_cat(choices_torch, dim=0).unsqueeze(
                    0)  # (1, T, tran.t, C, 1)
                choices_mask = data_util.multi_dim_padded_cat(choices_mask, dim=0).unsqueeze(0)  # (1, T, tran.t, C)
                if tran.t == 0:
                    trans_o.append(torch.zeros(1, len(words), 0, 0, 1).long())
                else:
                    trans_o.append(ibp.DiscreteChoiceTensorTrans(choices_torch, choices_mask, mask_torch))

            y_torch = torch.tensor(y, dtype=torch.float).view(1, 1)
            lengths_torch = torch.tensor(len(word_idxs)).view(1)
            examples.append(
                dict(x=x_torch, y=y_torch, mask=mask_torch, lengths=lengths_torch, trans_output=(trans_o, trans_phi)))
        return cls(raw_data, vocab, examples)

    @staticmethod
    def example_len(example):
        return example['x'].shape[1]

    @staticmethod
    def collate_examples(examples):
        """
        Turns a list of examples into a workable batch:
        """
        if len(examples) == 1:
            list_o, list_phi = examples[0]['trans_output']
            trans_o = []
            for o in list_o:
                if isinstance(o, ibp.DiscreteChoiceTensorTrans) and o.choice_mask.nelement() == 0:
                    trans_o.append(torch.zeros(1, examples[0]["lengths"], 0, 0, 1).long())
                else:
                    trans_o.append(o)
            return {'x': examples[0]["x"], 'trans_output': (trans_o, list_phi), 'y': examples[0]["y"],
                    'mask': examples[0]["mask"], 'lengths': examples[0]["lengths"]}
        B = len(examples)
        max_len = max(ex['x'].shape[1] for ex in examples)
        x_vals = []
        length_perturb = len(examples[0]['trans_output'][0])
        choice_mats = [[] for _ in range(length_perturb)]
        choice_masks = [[] for _ in range(length_perturb)]
        trans_phi = [[] for _ in range(length_perturb)]
        y = torch.zeros((B, 1))
        lengths = torch.zeros((B,), dtype=torch.long)
        masks = torch.zeros((B, max_len))
        for i, ex in enumerate(examples):
            x_vals.append(ex['x'])
            list_o, list_phi = ex['trans_output']
            # o and phi are two list with length of number of transformations
            for tran_id, (o, phi) in enumerate(zip(list_o, list_phi)):
                trans_phi[tran_id].append(phi)
                if isinstance(o, ibp.DiscreteChoiceTensorTrans):
                    choice_mats[tran_id].append(o.choice_mat)
                    choice_masks[tran_id].append(o.choice_mask)
                else:
                    choice_mats[tran_id].append(None)
                    choice_masks[tran_id].append(None)
            cur_len = ex['x'].shape[1]
            masks[i, :cur_len] = list_o[0].sequence_mask[0]
            y[i, 0] = ex['y']
            lengths[i] = ex['lengths'][0]
        x_vals = data_util.multi_dim_padded_cat(x_vals, 0).long()
        trans_o = []
        for tran_id in range(length_perturb):
            if choice_mats[tran_id][0] is None or all(x.nelement() == 0 for x in choice_masks[tran_id]):
                trans_o.append(torch.zeros(B, max_len, 0, 0, 1).long())
            else:
                trans_o.append(
                    ibp.DiscreteChoiceTensorTrans(data_util.multi_dim_padded_cat(choice_mats[tran_id], 0).long(),
                                                  data_util.multi_dim_padded_cat(choice_masks[tran_id], 0).long(),
                                                  masks))
            trans_phi[tran_id] = data_util.multi_dim_padded_cat(trans_phi[tran_id], 0)
        return {'x': x_vals, 'trans_output': (trans_o, trans_phi), 'y': y, 'mask': masks, 'lengths': lengths}


class TextClassificationTreeDataset(data_util.ProcessedDataset):
    """
    Dataset that holds processed example dicts
    """

    @classmethod
    def from_raw_data(cls, trees, vocab, tree_data_vocab=None, PAD_WORD=None, attack_surface=None, downsample_to=None,
                      downsample_shard=0, perturbation=None):
        assert tree_data_vocab is not None
        assert PAD_WORD is not None
        if downsample_to:
            trees = trees[downsample_shard * downsample_to:(downsample_shard + 1) * downsample_to]
        examples = []
        tree_data_vocab_key_list = list(tree_data_vocab.keys())
        for tree in trees:
            all_ids = [int(x) for x in tree.ndata['x'] if int(x) != PAD_WORD]
            all_words = [tree_data_vocab_key_list[x] for x in all_ids]
            all_i_words = [i for i, x in enumerate(tree.ndata['x']) if int(x) != PAD_WORD]
            choices = [[] for _ in tree.ndata['x']]
            words = [None] * len(tree.ndata['x'])
            if perturbation is not None:
                perturb = Perturbation(perturbation, all_words, tree_data_vocab, attack_surface=attack_surface)
                choices_ = perturb.get_output_for_baseline_final_state()
                for i, choice, w in zip(all_i_words, choices_, all_words):
                    words[i] = w
                    choices[i] = [x for x in choice if x == UNK or x in vocab]

            elif attack_surface is not None:
                raise AttributeError
            else:
                for i, w in zip(all_i_words, all_words):
                    words[i] = w
            if len(words) == 0:
                continue
            word_idxs = [vocab.get_index(w) for w in words]  # w must be in the vocab
            x_torch = torch.tensor(word_idxs).view(-1, 1)  # (T, d)
            if perturbation is not None:
                unk_mask = torch.tensor([0 if UNK in c_list else 1 for c_list in choices],
                                        dtype=torch.long)  # (T)
                choices_word_idxs = [
                    torch.tensor([vocab.get_index(c) for c in c_list if c != UNK], dtype=torch.long) for c_list in
                    choices
                ]
                # if any(0 in c.view(-1).tolist() for c in choices_word_idxs):
                #     raise ValueError("UNK tokens found")
                choices_torch = pad_sequence(choices_word_idxs, batch_first=True).unsqueeze(2)  # (T, C, 1)
                choices_mask = (choices_torch.squeeze(-1) != 0).long()  # (T, C)
            elif attack_surface is not None:
                raise AttributeError
            else:
                choices_torch = x_torch.view(-1, 1, 1)  # (T, 1, 1)
                choices_mask = torch.ones_like(x_torch.view(-1, 1))
                unk_mask = None

            mask_torch = torch.Tensor([1 if x is not None else 0 for x in words])
            if unk_mask is None:
                unk_mask = torch.ones(len(word_idxs))
            x_bounded = ibp.DiscreteChoiceTensorWithUNK(x_torch, choices_torch, choices_mask, mask_torch, unk_mask)
            examples.append(dict(x=x_bounded, mask=mask_torch, trees=[tree], rawx=all_words))
        return cls(trees, vocab, examples)

    @staticmethod
    def example_len(example):
        return example['x'].shape[1]

    @staticmethod
    def collate_examples(examples):
        """
        Turns a list of examples into a workable batch:
        """
        x_vals = []
        choice_mats = []
        choice_masks = []
        trees = []
        masks = []
        unk_masks = []
        for i, ex in enumerate(examples):
            x_vals.append(ex['x'].val)
            # (T, C, 1)
            choice_mats.append(ex['x'].choice_mat)
            # (T, C)
            choice_masks.append(ex['x'].choice_mask)
            cur_len = ex['x'].shape[0]
            masks.append(ex['x'].sequence_mask)
            unk_masks.append(ex['x'].unk_mask if ex['x'].unk_mask is not None else torch.ones(cur_len))
            trees.extend(ex['trees'])
        trees = dgl.batch(trees)
        x_vals = torch.cat(x_vals, 0)
        masks = torch.cat(masks, 0)
        unk_masks = torch.cat(unk_masks, 0)
        choice_mats = data_util.multi_dim_padded_cat(choice_mats, 0).long()
        choice_masks = data_util.multi_dim_padded_cat(choice_masks, 0).long()
        return {'x': ibp.DiscreteChoiceTensorWithUNK(x_vals, choice_mats, choice_masks, masks, unk_masks),
                'trees': trees, 'mask': masks, 'y': trees.ndata['y']}

    @staticmethod
    def collate_examples_adv(examples):
        return examples


class ToyClassificationDataset(TextClassificationDataset):
    """
    Dataset that holds a toy sentiment classification data
    """
    VOCAB_LIST = [
        'cat', 'dog', 'fish', 'tiger', 'chicken',
        'hamster', 'bear', 'lion', 'dragon', 'horse',
        'monkey', 'goat', 'sheep', 'goose', 'duck']

    @classmethod
    def get_raw_data(cls, ignore_dir, data_size=5000, max_len=10, *args, **kwargs):
        data = []
        for t in range(data_size):
            seq_len = random.randint(3, max_len)
            words = [random.sample(cls.VOCAB_LIST, 1)[0] for i in range(seq_len - 1)]
            if random.random() > 0.5:
                words.append(words[0])
                y = 1
            else:
                other_words = list(cls.VOCAB_LIST)
                other_words.remove(words[0])
                words.append(random.sample(other_words, 1)[0])
                y = 0
            data.append((' '.join(words), y))
        num_train = int(round(data_size * 0.8))
        train_data = data[:num_train]
        dev_data = data[num_train:]
        print(dev_data[:10])
        return RawClassificationDataset(train_data, dev_data)


class ToyClassificationAttackSurface(attacks.AttackSurface):
    """Attack surface for ToyClassificationDataset."""

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list

    def get_swaps(self, words):
        swaps = []
        s = ' '.join(words)
        for i in range(len(words)):
            if i == 0 or i == len(words) - 1:
                swaps.append([])
            else:
                swaps.append(self.vocab_list)
        return swaps


class IMDBDataset(TextClassificationDataset):
    """
    Dataset that holds the IMDB sentiment classification data
    """

    @classmethod
    def read_text(cls, imdb_dir, split):
        if split == 'test':
            subdir = 'test'
        else:
            subdir = 'train'
        with open(os.path.join(imdb_dir, subdir, 'imdb_%s_files.txt' % split)) as f:
            filenames = [line.strip() for line in f]
        data = []
        num_words = 0
        for fn in tqdm(filenames):
            label = 1 if fn.startswith('pos') else 0
            with open(os.path.join(imdb_dir, subdir, fn)) as f:
                x_raw = f.readlines()[0].strip().replace('<br />', ' ')
                x_toks = word_tokenize(x_raw)
                num_words += len(x_toks)
                data.append((' '.join(x_toks), label))
        num_pos = sum(y for x, y in data)
        num_neg = sum(1 - y for x, y in data)
        avg_words = num_words / len(data)
        print('Read %d examples (+%d, -%d), average length %d words' % (
            len(data), num_pos, num_neg, avg_words))
        return data

    @classmethod
    def get_raw_data(cls, imdb_dir, test=False):
        train_data = cls.read_text(imdb_dir, 'train')
        if test:
            dev_data = cls.read_text(imdb_dir, 'test')
        else:
            dev_data = cls.read_text(imdb_dir, 'dev')
        return RawClassificationDataset(train_data, dev_data)


class SST2Dataset(TextClassificationDataset):
    """
    Dataset that holds the IMDB sentiment classification data
    """

    @classmethod
    def read_text(cls, sst2_dir, split):
        import tensorflow_datasets as tfds
        def prepare_ds(ds):
            data = []
            num_pos = 0
            num_neg = 0
            num_words = 0
            for features in tfds.as_numpy(ds):
                sentence, label = features["sentence"], features["label"]
                tokens = word_tokenize(sentence.decode('UTF-8'))
                data.append((' '.join(tokens), label))
                num_pos += label == 1
                num_neg += label == 0
                num_words += len(tokens)

            avg_words = num_words / len(data)
            print('Read %d examples (+%d, -%d), average length %d words' % (
                len(data), num_pos, num_neg, avg_words))
            return data

        def prepare_test_ds(ds):
            data = []
            num_pos = 0
            num_neg = 0
            num_words = 0
            for features in ds:
                features = features.strip()
                sentence, label = features[2:], features[:1]
                label = int(label)
                tokens = word_tokenize(sentence)
                data.append((' '.join(tokens), label))
                num_pos += label == 1
                num_neg += label == 0
                num_words += len(tokens)

            avg_words = num_words / len(data)
            print('Read %d examples (+%d, -%d), average length %d words' % (
                len(data), num_pos, num_neg, avg_words))
            return data

        if split == "train":
            ds_train = tfds.load(name="glue/sst2", split="train", shuffle_files=False)
            return prepare_ds(ds_train)
        elif split == "test":
            return prepare_test_ds(open(os.path.join(sst2_dir, "sst2test.txt")).readlines())
        else:
            ds_val = tfds.load(name="glue/sst2", split="validation", shuffle_files=False)
            return prepare_ds(ds_val)

    @classmethod
    def get_raw_data(cls, sst2_dir, test=False):
        train_data = cls.read_text(sst2_dir, 'train')
        if test:
            dev_data = cls.read_text(sst2_dir, 'test')
        else:
            dev_data = cls.read_text(sst2_dir, 'dev')
        return RawClassificationDataset(train_data, dev_data)


class DataAugmenter(data_util.DataAugmenter):
    def augment(self, dataset):
        new_examples = []
        for ex in tqdm(dataset.examples):
            new_examples.append(ex)
            x_orig = ex['x']  # (1, T, 1)
            choices = []
            for i in range(x_orig.shape[1]):
                cur_choices = torch.masked_select(
                    x_orig.choice_mat[0, i, :, 0], x_orig.choice_mask[0, i, :].type(torch.uint8))
                choices.append(cur_choices)
            for t in range(self.augment_by):
                x_word_id = [choices[i][random.choice(range(len(choices[i])))] for i in range(len(choices))]
                x_new = torch.stack([id for i, id in enumerate(x_word_id)
                                     if random.rand() < 0.5 or x_orig.unk_mask[i] > 0]).view(1, -1, 1)
                x_bounded = ibp.DiscreteChoiceTensorWithUNK(
                    x_new, x_orig.choice_mat, x_orig.choice_mask, x_orig.sequence_mask, torch.ones_like(x_new))
                ex_new = dict(ex)
                ex_new['x'] = x_bounded
                new_examples.append(ex_new)
        return TextClassificationDataset(None, dataset.vocab, new_examples)
