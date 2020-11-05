"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075
"""
import time
import itertools
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

import ibp
from perturbation import Perturbation
from text_classification import TextClassificationTreeDataset
import data_util


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox['h']))
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 cell_type='nary',
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        cell = TreeLSTMCell if cell_type == 'nary' else ChildSumTreeLSTMCell
        self.cell = cell(x_size, h_size)

    def forward(self, batch, g, h, c):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        g : dgl.DGLGraph
            Tree for computation.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # feed embedding
        embeds = self.embedding(batch['x'].val.squeeze(-1))
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) * batch['mask'].float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        logits = self.linear(h)
        return logits


def add(dictionary, key_name, value):
    if isinstance(value, ibp.IntervalBoundedTensor):
        dictionary[key_name + "val"] = value.val
        dictionary[key_name + "lb"] = value.lb
        dictionary[key_name + "ub"] = value.ub
    else:
        dictionary[key_name] = value


def get(dictionary, key_name):
    if key_name in dictionary:
        return dictionary[key_name]
    else:
        return ibp.IntervalBoundedTensor(dictionary[key_name + "val"], dictionary[key_name + "lb"],
                                         dictionary[key_name + "ub"])


class TreeLSTMCellDP(nn.Module):
    def __init__(self, x_size, h_size, deltas_p1, device):
        super(TreeLSTMCellDP, self).__init__()
        self.h_size = h_size
        self.deltas_p1 = deltas_p1
        self.device = device
        self.deltas_p1_ranges = [range(x) for x in deltas_p1]
        self.W_iou = ibp.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = ibp.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = ibp.Linear(2 * h_size, 2 * h_size)

    def message_func_dp(self, edges):
        keys = ['hval', 'cval', 'hlb', 'clb', 'hub', 'cub', "unk_mask"]
        ret = {}
        for key in keys:
            ret[key] = edges.src[key]
        return ret

    def reduce_func_dp(self, nodes):
        h = get(nodes.mailbox, "h")  # (n, 2, Del, Ins, Sub, d)
        c = get(nodes.mailbox, "c")
        unk_mask = nodes.mailbox["unk_mask"]  # (n, 2)
        # if both children can be deleted, then the parent can be deleted
        new_unk_mask = th.where((unk_mask[:, 0] > 0) & (unk_mask[:, 1] > 0), th.sum(unk_mask, 1),
                                th.zeros_like(unk_mask[:, 0]))

        def cal_h_cat_c(h_left, h_right, c_left, c_right):  # (n, d)
            h_cat = ibp.cat([h_left, h_right], dim=-1)
            f_cat = ibp.activation(th.sigmoid, self.U_f(h_cat))  # (n, 2 * d)
            c = f_cat[:, :self.h_size] * c_left + f_cat[:, self.h_size:] * c_right
            return self.U_iou(h_cat), c

        # (n, Del, Ins, Sub, 3 * d)
        new_iou = []
        # (n, Del, Ins, Sub, d)
        new_c = []
        for deltas in itertools.product(*self.deltas_p1_ranges):
            deltas_ranges = [range(x + 1) for x in deltas]
            piece = None
            for deltas_left in itertools.product(*deltas_ranges):
                deltas_right = [y - x for (x, y) in zip(deltas_left, deltas)]
                tmp = cal_h_cat_c(h[:, 0, deltas_left[0], deltas_left[1], deltas_left[2], :],
                                  h[:, 1, deltas_right[0], deltas_right[1], deltas_right[2], :],
                                  c[:, 0, deltas_left[0], deltas_left[1], deltas_left[2], :],
                                  c[:, 1, deltas_right[0], deltas_right[1], deltas_right[2], :])
                piece = ibp.merge(piece, tmp)
            new_iou.append(piece[0].unsqueeze(1))
            new_c.append(piece[1].unsqueeze(1))

        auxh = []
        auxc = []
        for delta0 in range(self.deltas_p1[0]):
            inds = th.arange(unk_mask.shape[0])
            clip_unk_mask0 = th.clamp(unk_mask[:, 0], 1, delta0).long()
            clip_unk_mask1 = th.clamp(unk_mask[:, 1], 1, delta0).long()
            aux_list = []
            for x in [h, c]:
                aux_l = ibp.where(((unk_mask[:, 0] > 0) & (unk_mask[:, 0] <= delta0)).view(-1, 1, 1, 1),
                                  # n, Ins, Sub, d
                                  x[inds, 1, delta0 - clip_unk_mask0],
                                  ibp.IntervalBoundedTensor.bottom(
                                      (h.shape[0], self.deltas_p1[1], self.deltas_p1[2], self.h_size),
                                      device=self.device))
                aux_r = ibp.where(((unk_mask[:, 1] > 0) & (unk_mask[:, 1] <= delta0)).view(-1, 1, 1, 1),
                                  # n, Ins, Sub, d
                                  x[inds, 0, delta0 - clip_unk_mask1],
                                  ibp.IntervalBoundedTensor.bottom(
                                      (h.shape[0], self.deltas_p1[1], self.deltas_p1[2], self.h_size),
                                      device=self.device))
                aux_list.append(aux_l.merge(aux_r))

            auxh.append(aux_list[0].unsqueeze(dim=1))
            auxc.append(aux_list[1].unsqueeze(dim=1))

        auxh = ibp.cat(auxh, dim=1)
        auxc = ibp.cat(auxc, dim=1)
        new_iou = ibp.cat(new_iou, dim=1).view(-1, self.deltas_p1[0], self.deltas_p1[1], self.deltas_p1[2],
                                               self.h_size * 3)
        new_c = ibp.cat(new_c, dim=1).view(-1, self.deltas_p1[0], self.deltas_p1[1], self.deltas_p1[2], self.h_size)
        ret = {}
        add(ret, "iou", new_iou)
        add(ret, "c", new_c)
        add(ret, "auxh", auxh)
        add(ret, "auxc", auxc)
        ret["unk_mask"] = new_unk_mask
        return ret

    def apply_node_func_dp(self, nodes):
        iou = get(nodes.data, 'iou') + self.b_iou  # (n, Del, Ins, Sub, d * 3)
        i, o, u = iou[:, :, :, :, :self.h_size], iou[:, :, :, :, self.h_size: self.h_size * 2], iou[:, :, :, :,
                                                                                                self.h_size * 2:]
        i = ibp.activation(th.sigmoid, i)
        o = ibp.activation(th.sigmoid, o)
        u = ibp.activation(th.tanh, u)
        c = i * u + get(nodes.data, 'c')
        h = o * ibp.activation(th.tanh, c)
        auxh = get(nodes.data, 'auxh')
        auxc = get(nodes.data, 'auxc')
        ret = {}
        add(ret, "h", h.merge(auxh))
        add(ret, "c", c.merge(auxc))
        ret["unk_mask"] = nodes.data["unk_mask"]
        return ret

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTMDP(nn.Module):
    def __init__(self,
                 device,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 cell_type='nary',
                 pretrained_emb=None,
                 no_wordvec_layer=False,
                 perturbation=None):
        super(TreeLSTMDP, self).__init__()
        assert perturbation is not None
        # TODO: to implement more discrete perturbation space
        self.device = device
        self.perturbation = perturbation
        self.deltas = Perturbation.str2deltas(perturbation)
        self.deltas_p1 = [delta + 1 for delta in self.deltas]
        self.x_size = x_size
        self.h_size = h_size
        self.embedding = ibp.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = ibp.Dropout(dropout)
        self.linear = ibp.Linear(h_size, num_classes)
        cell = TreeLSTMCellDP if cell_type == 'nary' else None
        if cell is None:
            raise NotImplementedError
        self.no_wordvec_layer = no_wordvec_layer
        if no_wordvec_layer:
            self.cell = cell(x_size, h_size, self.deltas_p1, device)
        else:
            self.linear_input = ibp.Linear(x_size, h_size)
            self.cell = cell(h_size, h_size, self.deltas_p1, device)

    def query(self, x, vocab, device, tree_data_vocab, PAD_WORD, return_bounds=False, attack_surface=None,
              perturbation=None):
        """Query the model on a Dataset.

        Args:
          x: a list of trees
          vocab: vocabulary
          device: torch device.
          tree_data_vocab: original vocabulary in the dataset
          PAD_WORD: PAD_WORD in the dataset

        Returns: list of logits of same length as |examples|.
        """
        dataset = TextClassificationTreeDataset.from_raw_data(x, vocab, tree_data_vocab=tree_data_vocab,
                                                              PAD_WORD=PAD_WORD, attack_surface=attack_surface,
                                                              perturbation=perturbation)
        data = dataset.get_loader(len(x))

        with th.no_grad():
            batch = data_util.dict_batch_to_device(next(iter(data)), device)
            g = batch['trees']
            root_ids = [i for i in range(g.number_of_nodes()) if g.out_degrees(i) == 0]
            target = batch['y'].long()[root_ids]
            n = g.number_of_nodes()
            h = th.zeros((n, self.h_size)).to(device)
            c = th.zeros((n, self.h_size)).to(device)

            logits = self.forward(batch, g, h, c, compute_bounds=return_bounds)[root_ids]
            if return_bounds:
                r = logits.ub.clone()
                inds = th.arange(r.shape[0], device=device).long()
                r[inds, target] = logits.lb[inds, target]
                return logits.val, r
            else:
                return logits

    def forward(self, batch, g, h, c, compute_bounds=True, cert_eps=1.0):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        g : dgl.DGLGraph
            Tree for computation.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # feed embedding
        if compute_bounds:
            x = batch['x']
        else:
            x = batch['x'].val
        mask = batch['mask'].unsqueeze(-1)

        x_vecs = self.embedding(x)  # n, d
        z_interval = None
        unk_mask = None
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # n, h
            if isinstance(x_vecs, ibp.DiscreteChoiceTensorWithUNK):
                x_interval = x_vecs.to_interval_bounded(eps=cert_eps)
                unk_mask = x_vecs.unk_mask.to(self.device)
                x_vecs = x_vecs.val
                z_interval = ibp.activation(F.relu, x_interval)  # n, h

            z = ibp.activation(F.relu, x_vecs)  # n, h
        else:
            if isinstance(x_vecs, ibp.DiscreteChoiceTensorWithUNK):
                x_interval = x_vecs.to_interval_bounded(eps=cert_eps)
                x_vecs = x_vecs.val
                z_interval = x_interval

            z = x_vecs

        if z_interval is None:  # bypass the ibp
            g.ndata['iou'] = self.cell.W_iou(self.dropout(z)) * mask.float()
            g.ndata['h'] = h
            g.ndata['c'] = c
            # propagate
            dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func,
                                apply_node_func=self.cell.apply_node_func)
            # compute logits
            h = self.dropout(g.ndata.pop('h'))
            logits = self.linear(h)
            return logits

        D = len(self.deltas)
        d = h.shape[-1]
        n = h.shape[0]

        def dup(x):
            return x.view([n] + [1] * D + [-1]).repeat(1, *self.deltas_p1, 1)

        unk_mask = (1 - unk_mask).long() if unk_mask is not None else th.zeros_like(mask).long()
        h = dup(h)
        auxh = ibp.IntervalBoundedTensor.bottom_like(h)
        h = ibp.IntervalBoundedTensor.point(h)
        c = dup(c)
        auxc = ibp.IntervalBoundedTensor.bottom_like(c)
        c = ibp.IntervalBoundedTensor.point(c)

        iou_x = dup(self.cell.W_iou(self.dropout(z)) * mask.float())  # (nodes, Del, Ins, Sub, d * 3)
        iou_x = ibp.IntervalBoundedTensor.point(iou_x)
        if self.deltas[2] > 0:  # if has Sub
            iou_ibp = self.cell.W_iou(self.dropout(z_interval)) * mask.float()  # (nodes, d * 3)
            iou_x[:, 0, 0, 1, :] = iou_ibp  # (nodes, d * 3)
        if self.deltas[1] > 0:  # if has Ins
            iou = iou_x.val[:, 0, 0, 0, :] + self.cell.b_iou  # (nodes, d * 3)
            i, o, u = iou[:, :d], iou[:, d: d * 2], iou[:, d * 2:]
            i = th.sigmoid(i)
            o = th.sigmoid(o)
            u = th.tanh(u)
            ci = i * u
            hi = o * th.tanh(ci)
            h_cat = hi.repeat(1, 2)  # (nodes, d * 2)
            f_cat = th.sigmoid(self.cell.U_f(h_cat))
            ci = (f_cat[:, :d] + f_cat[:, d:]) * ci * mask.float()
            iou = self.cell.U_iou(h_cat) * mask.float()
            iou = ibp.IntervalBoundedTensor.point(iou)
            iou_x[:, 0, 1, 0, :] = iou
            ci = ibp.IntervalBoundedTensor.point(ci)
            c[:, 0, 1, 0, :] = ci

        add(g.ndata, "iou", iou_x)
        add(g.ndata, "h", h)
        add(g.ndata, "c", c)
        add(g.ndata, "auxh", auxh)
        add(g.ndata, "auxc", auxc)
        # g.ndata["mask"] = mask
        g.ndata["unk_mask"] = unk_mask

        # propagate
        dgl.prop_nodes_topo(g, self.cell.message_func_dp, self.cell.reduce_func_dp,
                            apply_node_func=self.cell.apply_node_func_dp)
        # (n, *, d)
        h_extend = self.dropout(
            ibp.IntervalBoundedTensor(g.ndata.pop('hval'), g.ndata.pop('hlb'), g.ndata.pop('hub'))).view(n, -1, d)
        # merge h
        h = h_extend[:, 0, :]
        for i in range(1, h_extend.shape[1]):
            h = h.merge(h_extend[:, i, :])
        # compute logits
        logits = self.linear(h)
        return logits
