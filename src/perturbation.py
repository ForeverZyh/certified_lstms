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


class Trans1(Transformation):
    def __init__(self, ipt, delta):
        """
        Movie review transformation 1.
        this is / it is / this ’s / it ’s
        """
        super(Trans1, self).__init__(2, 2, ipt, delta)
        self.matched_set = {("this", "is"), ("it", "is"), ("this", "'s"), ("it", "'s")}

    def phi(self, start_pos):
        return (self.ipt[start_pos], self.ipt[start_pos + 1]) in self.matched_set

    def transformer(self, start_pos):
        ret = []
        if (self.ipt[start_pos], self.ipt[start_pos + 1]) in self.matched_set:
            for x in self.matched_set:
                if x != (self.ipt[start_pos], self.ipt[start_pos + 1]):
                    ret.append(x)
        return ret


class Trans2(Transformation):
    def __init__(self, ipt, delta):
        """
        Movie review transformation 2.
        the/this/a movie/film; the/these movies/films
        """
        super(Trans2, self).__init__(2, 2, ipt, delta)
        self.matched_set = [{("the", "movie"), ("the", "film"), ("this", "movie"), ("this", "film"), ("a", "movie"),
                             ("a", "film")},
                            {("the", "movies"), ("the", "films"), ("these", "movies"), ("these", "films")}]

    def phi(self, start_pos):
        return (self.ipt[start_pos], self.ipt[start_pos + 1]) in self.matched_set[0] or (
            self.ipt[start_pos], self.ipt[start_pos + 1]) in self.matched_set[1]

    def transformer(self, start_pos):
        ret = []
        for i in range(len(self.matched_set)):
            if (self.ipt[start_pos], self.ipt[start_pos + 1]) in self.matched_set[i]:
                for x in self.matched_set[i]:
                    if x != (self.ipt[start_pos], self.ipt[start_pos + 1]):
                        ret.append(x)
        return ret


class Trans3(Transformation):
    def __init__(self, ipt, delta):
        """
        Movie review transformation 3. one of the most -> the most; one of the xxxest -> the xxxest
        """
        super(Trans3, self).__init__(4, 2, ipt, delta)

    def phi(self, start_pos):
        return (self.ipt[start_pos], self.ipt[start_pos + 1], self.ipt[start_pos + 2], self.ipt[start_pos + 3]) == (
            "one", "of", "the", "most") or ((self.ipt[start_pos], self.ipt[start_pos + 1], self.ipt[start_pos + 2]) == (
            "one", "of", "the") and self.ipt[start_pos + 3][-3:] == "est")

    def transformer(self, start_pos):
        if (self.ipt[start_pos], self.ipt[start_pos + 1], self.ipt[start_pos + 2], self.ipt[start_pos + 3]) == (
                "one", "of", "the", "most") or (
                (self.ipt[start_pos], self.ipt[start_pos + 1], self.ipt[start_pos + 2]) == ("one", "of", "the") and
                self.ipt[start_pos + 3][-3:] == "est"):
            return [(self.ipt[start_pos + 2], self.ipt[start_pos + 3])]
        else:
            return []


class Trans4(Transformation):
    def __init__(self, ipt, delta):
        """
        Movie review transformation 4. !->!!, ?->??
        """
        super(Trans4, self).__init__(1, 2, ipt, delta)
        self.puncs = ["!", "?"]

    def phi(self, start_pos):
        return self.ipt[start_pos] in self.puncs

    def transformer(self, start_pos):
        if self.ipt[start_pos] in self.puncs:
            return [(self.ipt[start_pos], self.ipt[start_pos])]
        else:
            return []


class Perturbation:
    Del_idx = 0
    Ins_idx = 1
    Sub_idx = 2

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
            elif tran in [Trans1, Trans2, Trans3, Trans4]:
                self.trans.append(tran(self.ipt, delta))
            else:
                raise NotImplementedError

    @staticmethod
    def dumy_perturbation(perturb_str):
        perturb = eval(perturb_str)
        trans = []
        deltas = []
        for tran, delta in perturb:
            deltas.append(delta)
            # TODO: a better way would be decouple the s, t attribute from the concrete ipt.
            if tran == Sub:
                trans.append(Sub([], delta, []))
            elif tran == Del:
                trans.append(Del([], delta))
            elif tran == Ins:
                trans.append(Ins([], delta))
            elif tran in [Trans1, Trans2, Trans3, Trans4]:
                trans.append(tran([], delta))
            else:
                raise NotImplementedError
        return trans, deltas

    def get_output_for_baseline_final_state(self):
        ret = [[] for _ in range(len(self.ipt))]
        for i in range(len(self.ipt)):
            ret[i] = set()
            for tran in self.trans:
                if isinstance(tran, Sub) and tran.phi(i):
                    choices = tran.transformer(i)
                    for choice in choices:
                        ret[i].add(choice[0])
            if self.has_del:
                for tran in self.trans:
                    if isinstance(tran, Del) and tran.phi(i):
                        ret[i].add(UNK)  # add dummy word for Del
            if self.ipt[i] in ret[i]:
                ret[i] = list(ret[i])
            else:
                ret[i] = [self.ipt[i]] + list(ret[i])

            # TODO: This is a workaround for Dup, i.e., duplicate a previous word. One can have a general Ins, or even
            # TODO: conditional Ins, but that will be a completely different implementation.
            # TODO: This workaround is efficient and specially designed for Dup. However, a more general implementation
            # TODO: will suffer from heavily CPU/GPU switch, thus is much less efficient than the workaround.
            # if self.has_ins and i % 2 == 1:
            #     for tran in self.trans:
            #         if isinstance(tran, Ins) and tran.phi(i // 2):
            #             ret[i].add(self.ipt[i // 2])  # add word for Ins

        return ret

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
