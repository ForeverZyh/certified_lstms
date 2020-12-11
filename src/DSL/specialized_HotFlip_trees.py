import numpy as np
import copy

from DSL.transformation import Del, Ins, Sub, Transformation
from utils import Beam, cons_tree


class HotFlipAttackTree:
    def __init__(self, perturbation: list, use_random_aug=False):
        """
        General HotFlip Attack for trees (on TreeLSTM)
        :param list perturbation: A perturbation space specified in the DSL.
        For example, [(Sub(), 2), (Del(), 1)] means at most 2 Sub string transformations and at most 1 Del string
        transformation. Sub and Del are default string transformations (see transformation.py).

        :Classifier Capacity: Gradient
        """
        self.trans2id = {Del: 0, Ins: 1, Sub: 2}
        try:
            for (a, b) in perturbation:
                # we currently only support three transformations for trees.
                assert isinstance(a, Del) or isinstance(a, Ins) or isinstance(a, Sub)
                assert isinstance(b, int)
        except:
            raise AttributeError("param perturbation %s is not in the correct form. "
                                 "Notice that we currently only support three transformations for trees" % str(
                perturbation))

        self.perturbation = perturbation
        self.use_random_aug = use_random_aug

    def gen_adv(self, model, tree, x: list, top_n: int, dataset_vocab, return_score=False):
        """
        Beam search for the perturbation space. The order of beam search is the same in the perturbation DSL.
        Some adversarial attack tries to rearrange the order of beam search for better performance.
        TODO: the order of beam search can be rearranged for better performance.
        :param model: the victim model, which has to support a method get_grad.
        :param tree: the target tree
        :param x: a list of input tokens.
        :param top_n: maximum number of adversarial candidates given the perturbation space.
        :param dataset_vocab: the vocabulary in the dataset, not the vocabulary used in training
        :param return_score: whether return the score as a list [(sen, score)], default False, i.e., return [sen]
        :return: a list of adversarial examples
        """

        try:
            model.get_grad
        except:
            raise AttributeError("The victim model does not support get_grad method.")

        meta_data = (x, tree, dataset_vocab)
        candidate = CandidateTree(tree, x, 0 if not self.use_random_aug else np.random.random())
        candidates = Beam(top_n)
        candidates.add(candidate, candidate.score)
        for (tran, delta) in self.perturbation:
            possible_pos = tran.get_pos(x)  # get a list of possible positions
            perturbed_set = set()  # restore the perturbed candidates to eliminate duplications
            for _ in range(delta):
                # keep the old candidates because we will change candidates in the following loop
                old_candidates = candidates.check_balance()
                for (candidate, _) in old_candidates:
                    if candidate not in perturbed_set:
                        if len(x) > 0:
                            if self.use_random_aug:
                                candidate.try_all_pos(meta_data, possible_pos, tran, None, model, candidates)
                            else:
                                candidate.try_all_pos(meta_data, possible_pos, tran,
                                                      model.get_grad(candidate.tree, isinstance(tran, Ins)),
                                                      model, candidates)
                        perturbed_set.add(candidate)

        ret = candidates.check_balance()
        if return_score:
            return [(x.tree, x.score) for (x, _) in ret]
        else:
            return [x.tree for (x, _) in ret]


class CandidateTree:
    def __init__(self, tree, x, score, map_ori2x=None, trans_on_pos=None, syns_on_pos=None):
        """
        Init a candidate
        :param tree: tree
        :param x: input tokens
        :param score: score of adversarial attack, the larger the better
        :param map_ori2x: position mapping from the original input to transformed input
        :param trans_on_pos: transformation on positions, len(trans_on_pos) = len(original input),
        [1 -> Del, 2 -> Dup, 3 -> Sub, 0 -> Nothing]
        :param syns_on_pos: synonyms on each position. len(syns_on_pos) = len(original input).
        """
        self.tree = tree
        self.x = x
        self.score = score
        if map_ori2x is None:
            self.map_ori2x = [_ for _ in range(len(x))]
        else:
            self.map_ori2x = map_ori2x
        if trans_on_pos is None:
            self.trans_on_pos = [0] * len(x)
        else:
            self.trans_on_pos = trans_on_pos
        if syns_on_pos is None:
            self.syns_on_pos = [None] * len(x)
        else:
            self.syns_on_pos = syns_on_pos

    def __lt__(self, other):
        return self.x < other.x

    def try_all_pos(self, meta_data, pos: list, tran: Transformation, gradients, victim_model, candidates: Beam):
        """
        Try all possible positions for trans
        :param meta_data: (ori, old_tree, vocab) original input tokens, original tree, dataset vocab (not the vocab in
        training). Difference: dataset vocab contains out of vocab (OOV) words, while vocab in training map OOV to UNK.
        :param pos: possible positions, a list of (start_pos, end_pos)
        :param tran: the target transformation
        :param gradients: the gradients tensor with respect to self.x
        :param victim_model: the victim model
        :param candidates: a beam of candidates, will be modified by this methods
        :return: None
        """

        get_embed = victim_model.get_embed
        ori, old_tree, vocab = meta_data
        for (start_pos_ori, end_pos_ori) in pos:
            if all(self.map_ori2x[i] is not None for i in range(start_pos_ori, end_pos_ori)):
                start_pos_x = self.map_ori2x[start_pos_ori]
                # notice that self.map_ori2x[end_pos] can be None, we need to calculate from self.map_ori2x[end_pos - 1]
                end_pos_x = self.map_ori2x[end_pos_ori - 1] + 1

                for new_x in tran.transformer(self.x, start_pos_x, end_pos_x):
                    delta_len = len(new_x) - len(self.x)
                    if isinstance(tran, Del):
                        new_trans_on_pos = self.trans_on_pos[:start_pos_ori] + [1] + self.trans_on_pos[end_pos_ori:]
                        new_syns_on_pos = copy.copy(self.syns_on_pos)
                        if gradients is not None:
                            old_embedding = get_embed([self.x[start_pos_x]])[0]  # (dim)
                            new_score = self.score + np.sum((0 - old_embedding) * gradients[start_pos_x])
                        else:
                            new_score = np.random.random()
                    elif isinstance(tran, Ins):
                        new_trans_on_pos = self.trans_on_pos[:start_pos_ori] + [2] + self.trans_on_pos[end_pos_ori:]
                        new_syns_on_pos = copy.copy(self.syns_on_pos)
                        old_embedding = get_embed([self.x[start_pos_x]])  # (1, dim)
                        if gradients is not None:
                            ioux_grads, c_grads = gradients
                            delta_ioux, delta_c = victim_model.model.cal_delta_Ins(old_embedding)
                            new_score = self.score + np.sum(ioux_grads[start_pos_x] * delta_ioux) + np.sum(
                                c_grads[start_pos_x] * delta_c)
                        else:
                            new_score = np.random.random()
                    elif isinstance(tran, Sub):
                        if vocab.get(new_x[start_pos_x], -1) == -1:
                            continue
                        new_trans_on_pos = self.trans_on_pos[:start_pos_ori] + [3] + self.trans_on_pos[end_pos_ori:]
                        new_syns_on_pos = self.syns_on_pos[:start_pos_ori] + [new_x[start_pos_x]] + self.syns_on_pos[
                                                                                                    end_pos_ori:]
                        if gradients is not None:
                            old_embedding = get_embed([self.x[start_pos_x]])[0]  # (dim)
                            new_embedding = get_embed([new_x[start_pos_x]])[0]  # (dim)
                            new_score = self.score + np.sum((new_embedding - old_embedding) * gradients[start_pos_x])
                        else:
                            new_score = np.random.random()
                    else:
                        raise NotImplementedError
                    new_tree = cons_tree(ori, new_trans_on_pos, new_syns_on_pos, old_tree, vocab)
                    new_map_ori2x = self.map_ori2x[:start_pos_ori] + [None] * (end_pos_ori - start_pos_ori) + [
                        p if p is None else p + delta_len for p in self.map_ori2x[end_pos_ori:]]
                    new_candidate = CandidateTree(new_tree, new_x, new_score, new_map_ori2x, new_trans_on_pos,
                                                  new_syns_on_pos)
                    candidates.add(new_candidate, new_score)
