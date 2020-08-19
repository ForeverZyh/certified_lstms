from abc import ABC, abstractmethod
import copy
from functools import partial

UNK = "_UNK_"
class Transformation(ABC):
    def __init__(self, s, t, ipt, delta):
        """
        :param s: the input length
        :param t: the output length
        :param ipt: the input string
        :param delta: can be applied delta times
        """
        self.s = s
        self.t = t
        self.ipt = ipt
        self.delta = delta
        super().__init__()

    @abstractmethod
    def phi(self, start_pos):
        """
        :param start_pos: the start position
        :return: predicate for a segment ipt[start_pos:start_pos + self.s] of input
        """
        pass

    @abstractmethod
    def transformer(self, start_pos):
        """
        :param start_pos: the start position
        :return: transformer for a segment ipt[start_pos:start_pos + self.t] of input
        """
        pass

    def gen_output_for_dp(self):
        """
        :return: a list of lists. Each list is in the form of [choice_0, choice_1, ..., ], where each choice_i is a
        tuple and |choice_i| = self.t.
        """
        ret = [[] for _ in range(len(self.ipt))]
        for i in range(len(self.ipt) - self.s + 1):
            if self.phi(i):
                ret[i] = self.transformer(i)

        return ret


class Sub(Transformation):
    def __init__(self, ipt, delta, swaps):
        super(Sub, self).__init__(1, 1, ipt, delta)
        self.swaps = swaps

    def phi(self, start_pos):
        return start_pos < len(self.ipt) and len(self.swaps[start_pos]) > 0

    def transformer(self, start_pos):
        return [(swap,) for swap in self.swaps[start_pos]]


class Ins(Transformation):
    def __init__(self, ipt, delta):
        super(Ins, self).__init__(1, 2, ipt, delta)

    def phi(self, start_pos):
        return start_pos < len(self.ipt)

    def transformer(self, start_pos):
        return [(self.ipt[start_pos], self.ipt[start_pos])]


class Del(Transformation):
    def __init__(self, ipt, delta, stop_words=None):
        super(Del, self).__init__(1, 0, ipt, delta)
        if stop_words is None:
            self.stop_words = {"a", "and", "the", "of", "to"}
        else:
            self.stop_words = stop_words

    def phi(self, start_pos):
        return start_pos < len(self.ipt) and self.ipt[start_pos] in self.stop_words

    def transformer(self, start_pos):
        return [()]


class Perturbation:
    def __init__(self, trans, ipt, vocab, attack_surface=None, stop_words=None):
        trans = eval(trans)
        self.ipt = [w for w in ipt if w in vocab]
        self.trans = []
        self.has_del = False
        self.has_ins = False
        for tran, delta in trans:
            if tran == Sub:
                assert attack_surface is not None
                swaps = attack_surface.get_swaps(ipt)
                self.trans.append(Sub(self.ipt, delta, [swap for swap, w in zip(swaps, ipt) if w in vocab]))
            elif tran == Del:
                self.trans.append(Del(self.ipt, delta, stop_words=stop_words))
                self.has_del = delta > 0
            elif tran == Ins:
                self.trans.append(Ins(self.ipt, delta))
                self.has_ins = delta > 0
            else:
                raise NotImplementedError

    def get_output_for_baseline_final_state(self):
        ret = [set() for _ in range(len(self.ipt) * 2)]
        for i in range(len(ret)):
            if i % 2 == 0:
                ret[i].add(self.ipt[i // 2])
                for tran in self.trans:
                    if isinstance(tran, Sub) and tran.phi(i // 2):
                        choices = tran.transformer(i // 2)
                        for choice in choices:
                            ret[i].add(choice[0])
            else:
                ret[i].add(UNK)  # add dummy word for Ins
            if self.has_del and i % 2 == 0:
                for tran in self.trans:
                    if isinstance(tran, Del) and tran.phi(i // 2):
                        ret[i].add(UNK)  # add dummy word for Del
            if self.has_ins and i % 2 == 1:
                for tran in self.trans:
                    if isinstance(tran, Ins) and tran.phi(i // 2):
                        ret[i].add(self.ipt[i // 2])  # add word for Ins

        return [list(x) for x in ret if len(x) > 1 or (UNK not in x and len(x) == 1)]

    def get_output_for_baseline(self):
        ret = [set() for _ in self.ipt]
        applied = [0 for _ in self.trans]

        # maximal number of times the transformations have been applied before the current position

        def cal_lower_upper_applied(applied):
            pos_lower_bound = 0
            pos_upper_bound = 0
            for id_tran, tran in enumerate(self.trans):
                if tran.t > tran.s:
                    pos_upper_bound += (tran.t - tran.s) * applied[id_tran]
                elif tran.t < tran.s:
                    pos_lower_bound += (tran.t - tran.s) * applied[id_tran]

            return pos_lower_bound, pos_upper_bound

        for i in range(len(self.ipt)):
            pos_lower_bound, pos_upper_bound = cal_lower_upper_applied(applied)
            for pos_enum in range(pos_lower_bound, pos_upper_bound + 1):
                output_pos = i + pos_enum  # the current output position according to i
                if output_pos < 0: continue
                while output_pos >= len(ret):
                    ret.append(set())
                ret[output_pos].add(self.ipt[i])  # add the current word

            old_applied = copy.copy(applied)
            for id_tran, tran in enumerate(self.trans):
                if tran.phi(i):
                    choices = tran.transformer(i)
                    old_applied[id_tran] = min(old_applied[id_tran], tran.delta - 1)  # restrict to tran.delta - 1
                    pos_lower_bound, pos_upper_bound = cal_lower_upper_applied(old_applied)
                    old_applied[id_tran] = applied[id_tran]
                    for pos_enum in range(pos_lower_bound, pos_upper_bound + 1):
                        output_pos = i + pos_enum  # the current output position according to i
                        if output_pos < 0: continue
                        for choice in choices:
                            # choice is a tuple and |choice| = self.t.
                            while output_pos + len(choice) - 1 >= len(ret):
                                ret.append(set())
                            for j, x in enumerate(choice):
                                ret[output_pos + j].add(x)

                    applied[id_tran] = min(applied[id_tran] + 1, tran.delta)

        return [list(x) for x in ret]

    @staticmethod
    def str2deltas(perturbation):
        # return deltas for [Del, Ins, Sub]
        p = eval(perturbation)
        trans = [Del, Ins, Sub]
        deltas = [0, 0, 0]
        for tran, delta in p:
            idx = trans.index(tran)
            if idx != -1:
                deltas[idx] = delta
            else:
                raise NotImplementedError

        return deltas


def tests():
    sen = ["i", "a", "to", "boys", "you", "tuzi", "a", "y"]
    a = Perturbation("[(Ins, 2), (Del, 2)]", sen)
    print(a.get_output_for_baseline())
    print(a.get_output_for_baseline_final_state())


# tests()
