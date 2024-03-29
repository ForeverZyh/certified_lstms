from abc import ABC, abstractmethod
import os
import pickle

from nltk import pos_tag


class Transformation(ABC):
    def __init__(self):
        """
        A default init function.
        """
        super().__init__()

    @abstractmethod
    def get_pos(self, ipt):
        # get matched positions in input
        pass

    @abstractmethod
    def transformer(self, ipt, start_pos, end_pos):
        # transformer for a segment of input
        pass


class Sub(Transformation):
    def __init__(self, pddb_path, use_fewer_sub):
        """
        Init Sub transformation
        Sub substitutes one word with its synonym with a requirement that they have the same pos_tag.
        :param pddb_path: the path to PPDB file
        :param use_fewer_sub: if True, one word can only be substituted by one of its synonyms.
        """
        self.use_fewer_sub = use_fewer_sub
        self.synonym_dict = {}
        self.synonym_dict_pos_tag = {}
        pddb_files = [f for f in os.listdir(pddb_path) if
                      os.path.isfile(os.path.join(pddb_path, f)) and f[:4] == "ppdb"]
        if len(pddb_files) == 0:
            raise AttributeError("No PPDB files found in %s" % pddb_path)
        else:
            pddb_file = pddb_files[0]
            print("Using ", pddb_files, " ...")

        lines = open(os.path.join(pddb_path, pddb_file)).readlines()
        for line in lines:
            tmp = line.strip().split(" ||| ")
            pos_tag, x, y = tmp[0][1:-1], tmp[1], tmp[2]
            self.synonym_dict_add_str(x, y, pos_tag)
            self.synonym_dict_add_str(y, x, pos_tag)

        print("Compute synonym_dict success!")
        super(Sub, self).__init__()

    def synonym_dict_add_str(self, x, y, pos_tag):
        if x not in self.synonym_dict:
            self.synonym_dict[x] = [y]
            self.synonym_dict_pos_tag[x] = [pos_tag]
        elif not self.use_fewer_sub:
            self.synonym_dict[x].append(y)
            self.synonym_dict_pos_tag[x].append(pos_tag)

    def get_pos(self, ipt):
        ipt_pos_tag = pos_tag(ipt)
        ret = []
        for (start_pos, (x, pos_tagging)) in enumerate(ipt_pos_tag):
            if x in self.synonym_dict:
                if any(pos_tagging == t for t in self.synonym_dict_pos_tag[x]):
                    ret.append((start_pos, start_pos + 1))

        return ret

    def transformer(self, ipt, start_pos, end_pos):
        ipt_pos_tag = pos_tag(ipt)
        x = ipt[start_pos]
        pos_tagging = ipt_pos_tag[start_pos][1]
        for (w, t) in zip(self.synonym_dict[x], self.synonym_dict_pos_tag[x]):
            if t == pos_tagging:
                new_ipt = ipt[:start_pos] + [w] + ipt[end_pos:]
                yield new_ipt


class SubImdb(Transformation):
    def __init__(self):
        """
        Init Sub transformation
        Sub substitutes one word with its synonym.
        """
        pkl_file = open("data/imdb_perturbation_constraint_pca0.8_100.pkl", 'rb')
        perturb_pca = pickle.load(pkl_file)
        pkl_file.close()
        self.neighbors = perturb_pca
        super(SubImdb, self).__init__()

    def get_pos(self, ipt):
        ret = []
        for (start_pos, x) in enumerate(ipt):
            if x in self.neighbors:
                ret.append((start_pos, start_pos + 1))

        return ret

    def transformer(self, ipt, start_pos, end_pos):
        x = ipt[start_pos]
        for w in self.neighbors[x]["set"]:
            new_ipt = ipt[:start_pos] + [w] + ipt[end_pos:]
            yield new_ipt


class Del(Transformation):
    def __init__(self, stop_words=None):
        """
        Init Del transformation
        Del removes one stop word.
        :param stop_words: stop words to delete
        """
        if stop_words is None:
            stop_words = {"a", "and", "the", "of", "to"}
        self.stop_words = stop_words

        super(Del, self).__init__()

    def get_pos(self, ipt):
        ret = []
        for (start_pos, x) in enumerate(ipt):
            if x in self.stop_words:
                ret.append((start_pos, start_pos + 1))

        return ret

    def transformer(self, ipt, start_pos, end_pos):
        new_ipt = ipt[:start_pos] + ipt[end_pos:]
        yield new_ipt


class Ins(Transformation):
    def __init__(self):
        """
        Init Ins transformation
        Ins duplicates a word behind it.
        """
        super(Ins, self).__init__()

    def get_pos(self, ipt):
        return [(x, x + 1) for x in range(len(ipt))]

    def transformer(self, ipt, start_pos, end_pos):
        new_ipt = ipt[:start_pos] + [ipt[start_pos], ipt[start_pos]] + ipt[end_pos:]
        yield new_ipt


class Trans1(Transformation):
    def __init__(self):
        """
        Movie review transformation 1. this is / it is / this ’s / it ’s
        """
        super(Trans1, self).__init__()
        self.matched_set = {("this", "is"), ("it", "is"), ("this", "'s"), ("it", "'s")}

    def get_pos(self, ipt):
        return [(x, x + 2) for x in range(len(ipt) - 1) if (ipt[x], ipt[x + 1]) in self.matched_set]

    def transformer(self, ipt, start_pos, end_pos):
        if (ipt[start_pos], ipt[start_pos + 1]) in self.matched_set:
            for x in self.matched_set:
                if x != (ipt[start_pos], ipt[start_pos + 1]):
                    new_ipt = ipt[:start_pos] + list(x) + ipt[end_pos:]
                    yield new_ipt


class Trans2(Transformation):
    def __init__(self):
        """
        Movie review transformation 2.
        the/this/a movie/film; the/these movies/films
        """
        super(Trans2, self).__init__()
        self.matched_set = [{("the", "movie"), ("the", "film"), ("this", "movie"), ("this", "film"), ("a", "movie"),
                             ("a", "film")},
                            {("the", "movies"), ("the", "films"), ("these", "movies"), ("these", "films")}]

    def get_pos(self, ipt):
        return [(x, x + 2) for x in range(len(ipt) - 1) if
                (ipt[x], ipt[x + 1]) in self.matched_set[0] or (ipt[x], ipt[x + 1]) in self.matched_set[1]]

    def transformer(self, ipt, start_pos, end_pos):
        for i in range(len(self.matched_set)):
            if (ipt[start_pos], ipt[start_pos + 1]) in self.matched_set[i]:
                for x in self.matched_set[i]:
                    if x != (ipt[start_pos], ipt[start_pos + 1]):
                        new_ipt = ipt[:start_pos] + list(x) + ipt[end_pos:]
                        yield new_ipt


class Trans3(Transformation):
    def __init__(self):
        """
        Movie review transformation 3. one of the most -> the most; one of the xxxest -> the xxxest
        """
        super(Trans3, self).__init__()

    def get_pos(self, ipt):
        return [(x, x + 4) for x in range(len(ipt) - 3) if
                (ipt[x], ipt[x + 1], ipt[x + 2], ipt[x + 3]) == ("one", "of", "the", "most") or (
                        (ipt[x], ipt[x + 1], ipt[x + 2]) == ("one", "of", "the") and ipt[x + 3][-3:] == "est")]

    def transformer(self, ipt, start_pos, end_pos):
        if (ipt[start_pos], ipt[start_pos + 1], ipt[start_pos + 2], ipt[start_pos + 3]) == (
                "one", "of", "the", "most") or (
                (ipt[start_pos], ipt[start_pos + 1], ipt[start_pos + 2]) == ("one", "of", "the") and ipt[start_pos + 3][
                                                                                                     -3:] == "est"):
            new_ipt = ipt[:start_pos] + [ipt[start_pos + 2], ipt[start_pos + 3]] + ipt[end_pos:]
            yield new_ipt


class Trans4(Transformation):
    def __init__(self):
        """
        Movie review transformation 4. !->!!, ?->??
        """
        super(Trans4, self).__init__()
        self.puncs = ["!", "?"]

    def get_pos(self, ipt):
        return [(x, x + 1) for x in range(len(ipt)) if ipt[x] in self.puncs]

    def transformer(self, ipt, start_pos, end_pos):
        if ipt[start_pos] in self.puncs:
            new_ipt = ipt[:start_pos] + [ipt[start_pos], ipt[start_pos]] + ipt[end_pos:]
            yield new_ipt
