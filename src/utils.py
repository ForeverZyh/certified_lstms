import numpy as np
import keras
import keras.backend as K
import heapq
import copy
from multiprocessing import Pool, Process, SimpleQueue, Pipe

import dgl
import networkx as nx

inf = 1e10


class Multiprocessing:
    @staticmethod
    def work(fun, child_conn, args):
        ret = fun(args[0], child_conn, args[2])
        child_conn.send(("close", ret))

    @staticmethod
    def mapping(fun, args_list, processes, partial_to_loss):
        ans = [None] * len(args_list)
        pipes = []
        for batch_start in range(0, len(args_list), processes):
            ps = []
            for i in range(batch_start, min(batch_start + processes, len(args_list))):
                parent_conn, child_conn = Pipe()
                pipes.append(parent_conn)
                p = Process(target=Multiprocessing.work, args=(fun, child_conn, args_list[i]))
                p.start()
                ps.append(p)

            unfinished = len(ps)
            while unfinished > 0:
                for i in range(batch_start, min(batch_start + processes, len(args_list))):
                    if pipes[i] is not None:
                        s = pipes[i].recv()
                        if len(s) == 2 and s[0] == "close":
                            ans[i] = s[1]
                            pipes[i] = None
                            unfinished -= 1
                        else:
                            pipes[i].send(partial_to_loss(s, args_list[i][1]))

            for p in ps:
                p.join()

        return ans


class MultiprocessingWithoutPipe:
    @staticmethod
    def work(fun, num, q, args):
        np.random.seed(num)
        ret = fun(*args)
        q.put((num, ret))

    @staticmethod
    def mapping(fun, args_list, processes):
        ans = [None] * len(args_list)
        q = SimpleQueue()
        for batch_start in range(0, len(args_list), processes):
            ps = []
            for i in range(batch_start, min(batch_start + processes, len(args_list))):
                p = Process(target=MultiprocessingWithoutPipe.work, args=(fun, i, q, args_list[i]))
                p.start()
                ps.append(p)

            while not q.empty():
                num, ret = q.get()
                ans[num] = ret

            for p in ps:
                p.join()

        while not q.empty():
            num, ret = q.get()
            ans[num] = ret

        return ans


class Gradient(keras.layers.Layer):
    def __init__(self, y, **kwargs):
        super(Gradient, self).__init__(**kwargs)
        self.y = y

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Gradient, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        return K.gradients(self.y, x)[0]

    def compute_output_shape(self, input_shape):
        return input_shape


def tuple_set_union(ret0: tuple, ret1: tuple):
    if ret0 is None:
        return ret1
    if ret1 is None:
        return ret0
    ret = ()
    max_len = max(len(ret0), len(ret1))
    for i in range(max_len):
        if i >= len(ret0):
            r0 = [""]
        else:
            r0 = ret0[i]
        if i >= len(ret1):
            r1 = [""]
        else:
            r1 = ret1[i]
        ret += (tuple(set(r0).union(set(r1))),)

    return ret


class Beam:
    def __init__(self, budget):
        '''
        The queue are contain two elements: (data, score), while the score is the ranking key (descending)
        :param budget: the beam search budget
        '''
        self.budget = budget
        self.queue = []
        self.in_queue = {}

    def add(self, data, score):
        '''
        add (data, score) into queue
        :param data: candidate
        :param score: score of the candidate
        :return: True if added in, False otherwise
        '''
        if data in self.in_queue:  # if data is ready in the priority queue, we update the score in self.in_queue
            if self.in_queue[data] < score:
                self.in_queue[data] = score
                return True
            return False

        ret = True
        if len(self.queue) == self.budget:  # if size(queue) == budget, we need to remove one
            while True:
                a, b = heapq.heappop(self.queue)
                # the top of the priority queue may not be smallest, because it may contain new value in self.in_queue
                if a == self.in_queue[b]:  # if the value is not updated, then it is smallest
                    break  # remove (a, b)
                else:
                    heapq.heappush(self.queue,
                                   (self.in_queue[b], b))  # otherwise, update in the priority queue (lazily)
            del self.in_queue[b]  # remove (a, b) from self.in_queue
            if a > score:  # if the old (a, b) is better then new (score, data), we replace (score, data) with (a, b)
                score, data = a, b
                ret = False

        heapq.heappush(self.queue, (score, data))  # add (score, data)
        self.in_queue[data] = score  # update in self.in_queue
        return ret

    def extend(self, others):
        if isinstance(others, list):
            for data, score in others:
                self.add(data, score)
        else:
            assert False
            # for data, score in others.queue:
            #     self.add(data, score)

    def check_balance(self):
        ret = []
        for data in self.in_queue:
            ret.append((data, self.in_queue[data]))
        ret.sort(key=lambda x: -x[1])
        return ret

    def is_same(self, others: list):
        if len(others) != len(self.queue):
            return False
        others.sort(key=lambda x: -x[1])
        for i in range(len(others)):
            data, score = others[i]
            if data not in self.in_queue or self.in_queue[data] != score:
                return False

        return True


class UnorderedBeam:
    def __init__(self, budget):
        self.budget = budget
        self.queue = []

    def add(self, data):
        self.queue.append(data)

    def extend(self, others):
        if isinstance(others, list):
            self.queue.extend(others)
        else:
            assert False
            # for data, score in others.queue:
            #     self.add(data, score)

    def check_balance(self):
        ids = np.random.randint(0, len(self.queue), self.budget)
        ret = []
        for id in ids:
            ret.append(self.queue[id])
        return ret


class Dict:
    def __init__(self, char2id):
        self.char2id = char2id
        self.id2char = [" "] * len(char2id)
        for c in char2id:
            self.id2char[char2id[c]] = c

    def to_string(self, ids):
        return "".join([self.id2char[x] for x in ids])

    def to_ids(self, s):
        return np.array([self.char2id[c] for c in s])


def swap_pytorch(x, p1, p2):
    z = x[p1].clone()
    x[p1] = x[p2]
    x[p2] = z


def compute_adjacent_keys(dict_map):
    lines = open("./dataset/en.key").readlines()
    adjacent_keys = [[] for i in range(len(dict_map))]
    for line in lines:
        tmp = line.strip().split()
        ret = set(tmp[1:]).intersection(dict_map.keys())
        ids = []
        for x in ret:
            ids.append(dict_map[x])
        adjacent_keys[dict_map[tmp[0]]].extend(ids)
    return adjacent_keys


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
                    g.add_node(cid, x=PAD_WORD, y=int(old_tree.ndata['y'][node]),
                               mask=0)  # we duplicate the y label
                    g.add_node(left, x=word, y=int(old_tree.ndata['y'][node]), mask=1)
                    g.add_node(right, x=word, y=int(old_tree.ndata['y'][node]), mask=1)
                    g.add_edge(left, cid)
                    g.add_edge(right, cid)
                    sub_trees.append(cid)
                    continue
                elif phi[id] == 3:
                    word = vocab.get(f[id], PAD_WORD)
                else:
                    raise NotImplementedError

                g.add_node(cid, x=word, y=int(old_tree.ndata['y'][node]), mask=1)
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
            g.add_node(nid, x=PAD_WORD, y=int(old_tree.ndata['y'][old_u]), mask=0)
            for cid in sub_trees:
                g.add_edge(cid, nid)
            return nid

    # add root
    root = _rec_build(0)
    g.add_node(root, x=PAD_WORD, y=int(old_tree.ndata['y'][0]), mask=0)
    assert old_tree.out_degrees(0) == 0  # sanity check
    return dgl.from_networkx(g, node_attrs=['x', 'y', 'mask'])
