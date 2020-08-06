from abc import ABC, abstractmethod
import copy


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

    def get_output_for_baseline(self):
        """
        :return: a list of lists. Each list is in the form of [choice_0, choice_1, ..., ], where each choice_i is a
        word.
        Notice: the output list can be longer than the self.ipt, because the Ins transformation can add words.
        """
        ret = [set() for _ in self.ipt]
        applied = 0  # maximal number of times the transformation has been applied before the current position
        for i in range(len(self.ipt) - self.s + 1):
            for enum_applied in range(applied + 1):
                output_pos = i + enum_applied * (self.t - self.s)  # the current output position according to i
                while output_pos >= len(ret):
                    ret.append(set())
                ret[output_pos].add(self.ipt[i])  # add the current word

            if self.phi(i):
                choices = self.transformer(i)
                for enum_applied in range(min(applied + 1, self.delta)): # restrict to self.delta - 1
                    output_pos = i + enum_applied * (self.t - self.s)  # the current output position according to i
                    for choice in choices:
                        # choice is a tuple and |choice| = self.t.
                        while output_pos + len(choice) - 1 >= len(ret):
                            ret.append(set())
                        for j, x in enumerate(choice):
                            ret[output_pos + j].add(x)

                applied = min(applied + 1, self.delta)

        return [list(x) for x in ret]


class Sub(Transformation):
    def __init__(self, ipt, delta, attack_surface, vocab):
        ipt = [w for w in ipt if w in vocab]
        super(Sub, self).__init__(1, 1, ipt, delta)
        self.swaps = attack_surface.get_swaps(ipt)

    def phi(self, start_pos):
        return len(self.swaps[start_pos]) > 0

    def transformer(self, start_pos):
        return [(swap,) for swap in self.swaps[start_pos]]


class Ins(Transformation):
    def __init__(self, ipt, delta, vocab):
        super(Ins, self).__init__(1, 2, [w for w in ipt if w in vocab], delta)

    def phi(self, start_pos):
        return start_pos < len(self.ipt)

    def transformer(self, start_pos):
        return [(self.ipt[start_pos], self.ipt[start_pos])]


class Del(Transformation):
    def __init__(self, ipt, delta, vocab, stop_words=None):
        super(Del, self).__init__(1, 0, [w for w in ipt if w in vocab], delta)
        if stop_words is None:
            self.stop_words = {"a", "and", "the", "of", "to"}
        else:
            self.stop_words = stop_words

    def phi(self, start_pos):
        return start_pos < len(self.ipt) and self.ipt[start_pos] in self.stop_words

    def transformer(self, start_pos):
        return [()]


class Perturbation:
    def __init__(self, trans: list):
        self.trans = trans
        for tran in self.trans:
            assert isinstance(tran, Transformation)

    def get_output_for_baseline(self):
        ret = [set() for _ in self.trans[0].ipt]
        for tran in self.trans:
            tmp = tran.get_output_for_baseline()
            for x, y in zip(ret, tmp):
                x.update(y)

        return [list(x) for x in ret]


def tests():
    sen = ["i", "a", "to", "boys", "you", "tuzi", "a", "y"]
    a = Perturbation([Ins(sen, 2, set(sen)), Del(sen, 2, set(sen))])
    print(a.get_output_for_baseline())
