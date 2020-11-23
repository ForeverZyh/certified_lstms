"""IBP text classification model."""
import itertools
import glob
import json
import numpy as np
import os
import pickle
import random
import copy

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

LOSS_FUNC = nn.BCEWithLogitsLoss()
LOSS_FUNC_KEEP_DIM = nn.BCEWithLogitsLoss(reduction="none")
IMDB_DIR = 'data/aclImdb'
LM_FILE = 'data/lm_scores/imdb_all.txt'
COUNTER_FITTED_FILE = 'data/counter-fitted-vectors.txt'
SST2_DIR = 'data/sst2'


class AdversarialModel(nn.Module):
    def __init__(self):
        super(AdversarialModel, self).__init__()

    def query(self, x, vocab, device, return_bounds=False, attack_surface=None, perturbation=None):
        """Query the model on a Dataset.

        Args:
          x: a string or a list
          vocab: vocabulary
          device: torch device.

        Returns: list of logits of same length as |examples|.
        """
        if isinstance(x, str):
            dataset = TextClassificationDataset.from_raw_data(
                [(x, 0)], vocab, attack_surface=attack_surface, perturbation=perturbation)
            data = dataset.get_loader(1)
        else:
            dataset = TextClassificationDataset.from_raw_data(
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
        if pool == 'final' and bidirectional:
            raise AttributeError("bidirectional not available when pool='final'")
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


class HotFlipAdversary(Adversary):
    """An Adversary that exhaustively tries all allowed perturbations.

    Only practical for short sentences.
    """

    def __init__(self, victim_model, perturbation):
        super(HotFlipAdversary, self).__init__(None)
        self.victim_model = victim_model
        from DSL.transformation import Ins, Del, Sub
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

            words = x.split()
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

            # TODO: count the number
            print('ExhaustiveAdversary: "%s" -> %d options' % (x, is_correct_single))
            is_correct.append(is_correct_single)
            if is_correct_single:
                adv_exs.append([])

        return is_correct, adv_exs


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
                attack_surface=self.attack_surface, perturbation=self.perturbation)
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

            words = x.split()
            swaps = self.attack_surface.get_swaps(words)
            choices = [[s for s in cur_swaps if s in dataset.vocab] for w, cur_swaps in zip(words, swaps) if
                       w in dataset.vocab]
            words = [w for w in words if w in dataset.vocab]

            is_correct_single = True
            for batch_x in ExhaustiveAdversary.DelDupSubWord(*self.deltas, words, choices, batch_size=10):
                all_raw = [' '.join(x_new) for x_new in batch_x]
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
    def cons_tree(x, phi, f, old_tree, vocab):
        PAD_WORD = -1
        g = nx.DiGraph()
        old_xid = old_tree.ndata['x'].tolist()
        cnt = 0
        map_old_xid_x_id = [None] * len(old_xid)
        for i, id in enumerate(old_xid):
            if id != PAD_WORD:  # PAD_WORD
                map_old_xid_x_id[i] = cnt
                cnt += 1

        assert cnt == len(x)  # sanity check

        def _rec_build(old_u):
            in_nodes = old_tree.in_edges(old_u)[0]
            sub_trees = []
            for node in in_nodes:
                node = int(node)
                if old_tree.in_degrees(node) == 0:
                    # leaf node
                    cid = g.number_of_nodes()
                    id = map_old_xid_x_id[node]
                    if phi[id] == 0:
                        word = vocab.get(x[id], PAD_WORD)
                    elif phi[id] == 1:
                        continue
                    elif phi[id] == 2:
                        left = cid + 1
                        right = cid + 2
                        word = vocab.get(x[id], PAD_WORD)
                        g.add_node(cid, x=PAD_WORD, y=0, mask=0)  # we do not care about the y label
                        g.add_node(left, x=word, y=0, mask=1)  # we do not care about the y label
                        g.add_node(right, x=word, y=0, mask=1)  # we do not care about the y label
                        g.add_edge(left, cid)
                        g.add_edge(right, cid)
                        sub_trees.append(cid)
                        continue
                    elif phi[id] == 3:
                        word = vocab.get(f[id], PAD_WORD)
                    else:
                        raise NotImplementedError

                    g.add_node(cid, x=word, y=0, mask=1)  # we do not care about the y label
                    sub_trees.append(cid)
                else:
                    sub_tree = _rec_build(node)
                    if sub_tree is not None:
                        sub_trees.append(sub_tree)

            if len(sub_trees) == 0:
                return None
            elif len(sub_trees) == 1:
                return sub_trees[0]
            else:
                assert len(sub_trees) == 2  # sanity check
                nid = g.number_of_nodes()
                g.add_node(nid, x=PAD_WORD, y=0, mask=0)  # we do not care about the y label
                for cid in sub_trees:
                    g.add_edge(cid, nid)
                return nid

        # add root
        root = _rec_build(0)
        g.add_node(root, x=PAD_WORD, y=int(old_tree.ndata['y'][0]), mask=0)
        assert old_tree.out_degrees(0) == 0  # sanity check
        return dgl.from_networkx(g, node_attrs=['x', 'y', 'mask'])

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

                                    X.append(ExhaustiveAdversary.cons_tree(x, phi, f, tree, vocab))
                                    if len(X) == batch_size:
                                        yield X
                                        X = []

        if len(X) > 0:
            yield X


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
            words = x.split()
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
            words = x.split()
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
            if not isinstance(train_data, data_class):
                raise Exception("Cached dataset of wrong class: {}".format(type(train_data)))
        with open(os.path.join(opts.data_cache_dir, 'dev_data.pkl'), 'rb') as infile:
            dev_data = pickle.load(infile)
            if not isinstance(dev_data, data_class):
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
        else:
            attack_surface = attacks.WordSubstitutionAttackSurface.from_file(opts.neighbor_file)
        print('Reading dataset.')
        if opts.dataset == "Imdb":
            raw_data = data_class.get_raw_data(opts.imdb_dir, test=opts.test)
        elif opts.dataset == "SST2":
            raw_data = data_class.get_raw_data(opts.sst2_dir, test=opts.test)
        else:
            raise NotImplementedError
        word_set = raw_data.get_word_set(attack_surface, use_counter_vocab=not opts.use_a3t_settings)
        vocab, word_mat = vocabulary.Vocabulary.read_word_vecs(word_set, opts.glove_dir, opts.glove, device)
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
        for x, y in raw_data:
            all_words = [w.lower() for w in x.split()]
            ins_delta = 0
            if perturbation is not None:
                perturb = Perturbation(perturbation, all_words, vocab, attack_surface=attack_surface)
                ins_delta = perturb.deltas[perturb.Ins_idx]
                choices = perturb.get_output_for_baseline_final_state()
                choices = [[x for x in choice if x == UNK or x in vocab] for choice in choices]
                words = perturb.ipt
            elif attack_surface is not None:
                raise AttributeError
            else:
                words = [w for w in all_words if w in vocab]  # Delete UNK words
            if truncate_to:
                words = words[:truncate_to]
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
