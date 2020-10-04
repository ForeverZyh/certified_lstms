import argparse

from stanza.server import CoreNLPClient
from dgl.data.tree import SSTDataset
import torch as th
from zss import simple_distance, Node
import numpy as np
import xlwt
import dgl
import networkx as nx

from text_classification import TextClassificationTreeDataset, ExhaustiveAdversary
import vocabulary
import data_util
import attacks


def parsed_tree_to_zzs_tree(u):
    if len(u.child) == 1:
        return parsed_tree_to_zzs_tree(u.child[0])
    elif len(u.child) > 1:
        assert len(u.child) == 2
        node = Node("PAD")
        node.addkid(parsed_tree_to_zzs_tree(u.child[0]))
        node.addkid(parsed_tree_to_zzs_tree(u.child[1]))
        return node
    else:
        return Node(u.value)


def dgl_tree_to_zzs_tree(tree, vocab_key_list, u):
    if tree.in_degrees(u) == 0:
        return Node(vocab_key_list[tree.ndata['x'][u]])
    node = Node("PAD")
    in_nodes = tree.in_edges(u)[0]
    for in_node in in_nodes:
        in_node = int(in_node)
        node.addkid(dgl_tree_to_zzs_tree(tree, vocab_key_list, in_node))
    return node


def parsed_tree_to_dgl_tree(parsed_tree, vocab):
    PAD_WORD = -1
    g = nx.DiGraph()

    def _rec_build(u):
        if len(u.child) == 1:
            return _rec_build(u.child[0])
        elif len(u.child) > 1:
            assert len(u.child) == 2
            nid = g.number_of_nodes()
            g.add_node(nid, x=PAD_WORD, y=0)
            left = _rec_build(u.child[0])
            right = _rec_build(u.child[1])
            g.add_edge(left, nid)
            g.add_edge(right, nid)
            return nid
        else:
            cid = g.number_of_nodes()
            word = vocab.get(u.value, PAD_WORD)
            g.add_node(cid, x=word, y=0)
            return cid

    # add root
    root = _rec_build(parsed_tree)
    g.add_node(root, x=PAD_WORD)
    return dgl.from_networkx(g, node_attrs=['x', 'y'])


def main(args):
    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)

    trainset = SSTDataset(mode='train')
    trainset_vocab = trainset.vocab

    vocab, word_mat = vocabulary.Vocabulary.read_word_vecs_tree_lstm(trainset.vocab, args.glove_dir, "6B.50d", device)
    attack_surface = attacks.A3TWordSubstitutionAttackSurface.from_file(args.pddb_file, args.use_fewer_sub)
    trainset = TextClassificationTreeDataset.from_raw_data(trainset, vocab, tree_data_vocab=trainset.vocab,
                                                           PAD_WORD=trainset.PAD_WORD,
                                                           attack_surface=attack_surface,
                                                           perturbation=args.perturbation)

    def dfs_first_order(u):
        if len(u.child) == 1:
            ret = dfs_first_order(u.child[0])
        elif len(u.child) > 1:
            assert len(u.child) == 2
            ret = ["."]
            ret += dfs_first_order(u.child[0])
            ret += dfs_first_order(u.child[1])
        else:
            ret = [u.value]
        return ret

    def dfs_mid_order(u):
        ret = []
        if len(u.child) == 1:
            ret += dfs_mid_order(u.child[0])
        elif len(u.child) > 1:
            assert len(u.child) == 2
            ret += dfs_mid_order(u.child[0])
            ret.append(".")
            ret += dfs_mid_order(u.child[1])
        else:
            ret = [u.value]
        return ret

    # print(trainset.examples[0]["rawx"])
    properties = {
        'tokenize.whitespace': True,
        "parse.binaryTrees": True,
        "ssplit.eolonly": True
    }
    with CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse'],
            timeout=30000,
            memory='16G', properties=properties) as client:

        def get_similarity(text, g):
            ann = client.annotate(" ".join(text + ["\n"]))
            # get the first sentence
            sentence = ann.sentence[0]

            # get the constituency parse of the first sentence
            constituency_parse = sentence.binarizedParseTree.child[0]
            parsed_tree = parsed_tree_to_zzs_tree(constituency_parse)
            dgl_tree = dgl_tree_to_zzs_tree(g, list(trainset_vocab.keys()),
                                            [i for i in range(g.number_of_nodes()) if g.out_degrees(i) == 0][0])
            # print(parsed_tree, dgl_tree)
            return simple_distance(parsed_tree, dgl_tree), constituency_parse

        workbook = xlwt.Workbook(encoding='utf-8')

        for sheet_name, deltas in zip(["Del", "Ins", "Sub"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
            # id, perturbed id (0 for baseline), dist
            worksheet = workbook.add_sheet(sheet_name)
            worksheet.write(0, 0, label="id")
            worksheet.write(0, 1, label="perturbed id")
            worksheet.write(0, 2, label="dist")
            cnt = 1
            for (id, example) in enumerate(trainset.examples[:100]):
                text = example["rawx"]
                tree = example["trees"][0]
                baseline_dist, parsed_tree = get_similarity(text, tree)
                worksheet.write(cnt, 0, label=id)
                worksheet.write(cnt, 1, label=0)
                worksheet.write(cnt, 2, label=baseline_dist)
                cnt += 1

                swaps = attack_surface.get_swaps(text)
                choices = [[s for s in cur_swaps if s in trainset_vocab] for w, cur_swaps in zip(text, swaps) if
                           w in trainset_vocab]

                similarities = []
                # for (perturbed_text, perturbed_tree) in zip(ExhaustiveAdversary.DelDupSubWord(*deltas, text, choices,
                #                                                                               batch_size=1),
                #                                             ExhaustiveAdversary.DelDupSubTree(*deltas, text, tree,
                #                                                                               trainset_vocab, choices,
                #                                                                               batch_size=1)):
                for (perturbed_text, perturbed_tree) in zip(ExhaustiveAdversary.DelDupSubWord(*deltas, text, choices,
                                                                                              batch_size=1),
                                                            ExhaustiveAdversary.DelDupSubTree(*deltas, text,
                                                                                              parsed_tree_to_dgl_tree(
                                                                                                  parsed_tree,
                                                                                                  trainset_vocab),
                                                                                              trainset_vocab, choices,
                                                                                              batch_size=1)):
                    similarities.append(get_similarity(perturbed_text[0], perturbed_tree[0])[0])

                mean_sim = float(np.mean(similarities))
                # assert baseline_dist == similarities[-1]  # sanity check
                assert 0 == similarities[-1]  # sanity check
                for (i, sim) in enumerate(similarities[:-1]):  # the last one is the baseline
                    worksheet.write(cnt, 0, label=id)
                    worksheet.write(cnt, 1, label=i + 1)
                    worksheet.write(cnt, 2, label=sim)
                    cnt += 1

                print("baseline distance: %d\t average perturbed distance: %.2f" % (baseline_dist, mean_sim))

        workbook.save('sim_parsed.xls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TreeLSTM
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--child-sum', action='store_true')
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no-wordvec-layer', action='store_true', help="Don't apply linear transform to word vectors")

    # Adversary
    parser.add_argument('--use-fewer-sub', action='store_true', help='Use one substitution per word')
    parser.add_argument('--perturbation', type=str, default=None)
    parser.add_argument('--aug-perturbation', type=str, default=None)

    # Data and files
    parser.add_argument('--adv-only', action='store_true',
                        help='Only run the adversary against the model on the given evaluation set')
    parser.add_argument('--test', action='store_true', help='Evaluate on test set')
    parser.add_argument('--pddb-file', type=str, default=data_util.PDDB_FILE)
    parser.add_argument('--glove-dir', type=str, default=vocabulary.GLOVE_DIR)
    parser.add_argument('--downsample-to', type=int, default=None,
                        help='Downsample train and dev data to this many examples')
    parser.add_argument('--downsample-shard', type=int, default=0,
                        help='Downsample starting at this multiple of downsample_to')
    args = parser.parse_args()
    print(args)
    main(args)
