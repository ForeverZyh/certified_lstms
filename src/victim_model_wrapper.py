import torch
import numpy as np

from text_classification import TextClassificationDataset as text
import data_util


class ModelWrapper:
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device

    def get_embed(self, x, ret_len=None):
        """
        :param x: a list of tokens
        :param ret_len: the return length, if None, the length is equal to len(x). if len(x) < ret_len, we add padding
        :return: np array with shape (ret_len, word_vec_size)
        """
        x = torch.LongTensor([self.vocab.get_index(w) for w in x]).to(self.device)
        return self.model.get_embed(x, ret_len)

    def get_grad(self, x, y):
        """
        :param x: a list of tokens
        :param y: a label
        :return: np array with shape (len, word_vec_size)
        """
        dataset = text.from_raw_data([(" ".join(x), y)], self.vocab)
        data = dataset.get_loader(1)
        batch = data_util.dict_batch_to_device(next(iter(data)), self.device)
        gradients = self.model.get_grad(batch, torch.Tensor([[y]]))[0]
        # some of the UNK have been omitted, we need to add them back as zero gradients
        if len(gradients) == len(x):
            return gradients
        ret = np.zeros((len(x), gradients.shape[1]))
        pointer = 0
        for (i, w) in enumerate(x):
            if self.vocab.get_index(w) > 0:
                ret[i] = gradients[pointer]
                pointer += 1

        return ret


class TreeModelWrapper:
    def __init__(self, model, vocab, device, from_raw_data):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.from_raw_data = from_raw_data

    def get_embed(self, x, ret_len=None):
        """
        :param x: a list of tokens
        :param ret_len: the return length, if None, the length is equal to len(x). if len(x) < ret_len, we add padding
        :return: np array with shape (ret_len, word_vec_size)
        """
        x = torch.LongTensor([self.vocab.get_index(w) for w in x]).to(self.device)
        return self.model.get_embed(x, ret_len)

    def get_grad(self, tree, attack_Ins):
        """
        :param tree: a target tree
        :param attack_Ins: if attack the Ins transformation, if True, we will return additional grads (x_iou, h, c)
        :return: np array with shape (len, word_vec_size)
        """
        dataset = self.from_raw_data([tree], self.vocab)
        data = dataset.get_loader(1)
        batch = data_util.dict_batch_to_device(next(iter(data)), self.device)
        return self.model.get_grad(batch, attack_Ins)
