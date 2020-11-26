import sys
import argparse
import collections
import time
import os
import json
import glob
from functools import partial

from tqdm import tqdm
import numpy as np
import torch as th

th.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import networkx as nx
import dgl
from dgl.data.tree import SSTDataset

from tree_lstm import TreeLSTM, TreeLSTMDP
from text_classification import TextClassificationTreeDataset, num_correct_multi_classes, ExhaustiveAdversary, \
    HotFlipAdversary
import vocabulary
import data_util
import attacks
from victim_model_wrapper import TreeModelWrapper
from DSL.transformation import Ins, Del, Sub
from DSL.specialized_HotFlip_trees import HotFlipAttackTree

SSTBatch = collections.namedtuple('SSTBatch', ['graph', 'mask', 'wordid', 'label'])
num_classes = 5


def hotflip_aug(attack, examples, data, victim_model, device, trainset_vocab):
    trees = []
    for i, ex in enumerate(examples):
        g = ex['trees'][0]
        text = ex['rawx']
        trees.append(g)
        ans = attack.gen_adv(victim_model, g, text, 1, trainset_vocab)[0]
        trees.append(ans)

    dataset = victim_model.from_raw_data(trees, data.vocab)

    return data_util.dict_batch_to_device(next(iter(dataset.get_loader(len(trees)))), device)


def test(args, model, dataset, device, name, adversary=None, trainset_vocab=None, PAD_WORD=None, victim_model=None,
         attack=None):
    if args.adv_perturbation is None or victim_model is None:  # make sure the adv testing does not affect clean acc
        data = dataset.get_loader(args.batch_size)
    else:
        data = dataset.get_loader_adv(args.batch_size)
    model.eval()
    loss_func = CrossEntropyLoss()
    results = {
        'name': name,
        'num_total': 0,
        'num_correct': 0,
        'num_cert_correct': 0,
        'clean_acc': 0.0,
        'cert_acc': 0.0,
        'loss': 0.0
    }
    with th.no_grad():
        for batch in tqdm(data):
            # make sure the adv testing does not affect clean acc
            if args.adv_perturbation is not None and victim_model is not None:
                batch = hotflip_aug(attack, batch, dataset, victim_model, device, trainset_vocab)
            else:
                batch = data_util.dict_batch_to_device(batch, device)
            g = batch['trees']
            root_ids = [i for i in range(g.number_of_nodes()) if g.out_degrees(i) == 0]
            n = g.number_of_nodes()
            h = th.zeros((n, args.h_size)).to(device)
            c = th.zeros((n, args.h_size)).to(device)

            out = model(batch, g, h, c, cert_eps=1.0)
            results['loss'] += loss_func(out.val, batch['y']).item()
            num_correct, num_cert_correct = num_correct_multi_classes(out[root_ids], batch['y'][root_ids], num_classes)
            results["num_correct"] += num_correct
            results["num_cert_correct"] += num_cert_correct
            results['num_total'] += len(root_ids)
    results['clean_acc'] = 100 * results['num_correct'] / results['num_total']
    results['cert_acc'] = 100 * results['num_cert_correct'] / results['num_total']
    out_str = "  {name} loss = {loss:.2f}; accuracy: {num_correct}/{num_total} = {clean_acc:.2f}, certified {num_cert_correct}/{num_total} = {cert_acc:.2f}".format(
        **results)
    if adversary:
        adv_correct, adv_exs = adversary.run_tree(model, dataset, device, trainset_vocab, PAD_WORD, opts=args)
        results['num_adv_correct'] = sum(adv_correct)
        results['adv_acc'] = 100 * results['num_adv_correct'] / results['num_total']
        out_str += ', adversarial %d/%d = %.2f' % (
            results['num_adv_correct'], results['num_total'], results['adv_acc'])
    print(out_str)
    return results


def train(args, model, trainset, devset, device, trainset_vocab, victim_model=None, attack=None):
    if args.adv_perturbation is None:
        train_loader = trainset.get_loader(args.batch_size)
    else:
        train_loader = trainset.get_loader_adv(args.batch_size)

    params_ex_emb = [x for x in list(model.parameters()) if x.requires_grad and x.size(0) != len(trainset_vocab) + 1]
    params_emb = list(model.embedding.parameters())

    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    optimizer = optim.Adagrad([
        {'params': params_ex_emb, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': params_emb, 'lr': 0.1 * args.lr}])

    print('Training model')
    sys.stdout.flush()
    loss_func = CrossEntropyLoss()
    zero_stats = {'epoch': 0, 'clean_acc': 0.0, 'cert_acc': 0.0}
    num_epochs = args.epochs
    all_epoch_stats = {
        "loss": {"total": [],
                 "clean": [],
                 "cert": []},
        "cert": {"frac": [],
                 "eps": []},
        "acc": {
            "train": {
                "clean": [],
                "cert": []},
            "dev": {
                "clean": [],
                "cert": []},
            "best_dev": {
                "clean": [zero_stats],
                "cert": [zero_stats]}},
        "total_epochs": num_epochs,
    }
    # Linearly increase the weight of adversarial loss over all the epochs to end up at the final desired fraction
    cert_schedule = th.tensor(
        np.linspace(0, args.cert_frac, num_epochs - args.full_train_epochs - args.non_cert_train_epochs),
        dtype=th.float, device=device)
    eps_schedule = th.tensor(
        np.linspace(0, 1, num_epochs - args.full_train_epochs - args.non_cert_train_epochs), dtype=th.float,
        device=device)
    for t in range(num_epochs):
        model.train()
        if t < args.non_cert_train_epochs:
            cur_cert_frac = 0.0
            cur_cert_eps = 0.0
        else:
            cur_cert_frac = cert_schedule[t - args.non_cert_train_epochs] if t - args.non_cert_train_epochs < len(
                cert_schedule) else cert_schedule[-1]
            cur_cert_eps = eps_schedule[t - args.non_cert_train_epochs] if t - args.non_cert_train_epochs < len(
                eps_schedule) else eps_schedule[-1]
        epoch = {
            "total_loss": 0.0,
            "clean_loss": 0.0,
            "cert_loss": 0.0,
            "num_correct": 0,
            "num_cert_correct": 0,
            "num": 0,
            "clean_acc": 0,
            "cert_acc": 0,
            "dev": {},
            "best_dev": {},
            "cert_frac": cur_cert_frac if isinstance(cur_cert_frac, float) else cur_cert_frac.item(),
            "cert_eps": cur_cert_eps if isinstance(cur_cert_eps, float) else cur_cert_eps.item(),
            "epoch": t,
        }
        with tqdm(train_loader) as batch_loop:
            for batch_idx, batch in enumerate(batch_loop):
                if args.adv_perturbation is not None:
                    model.eval()
                    optimizer.zero_grad()
                    batch = hotflip_aug(attack, batch, trainset, victim_model, device, trainset_vocab)
                    model.train()
                else:
                    batch = data_util.dict_batch_to_device(batch, device)

                g = batch['trees']
                target = batch['y'].long()
                root_ids = [i for i in range(g.number_of_nodes()) if g.out_degrees(i) == 0]
                n = g.number_of_nodes()
                h = th.zeros((n, args.h_size)).to(device)
                c = th.zeros((n, args.h_size)).to(device)

                optimizer.zero_grad()
                if cur_cert_frac > 0.0:
                    out = model(batch, g, h, c, cert_eps=cur_cert_eps)
                    logits = out.val
                    loss = loss_func(logits, batch['y'])
                    epoch["clean_loss"] += loss.item()
                    r = out.ub.clone()
                    inds = th.arange(r.shape[0], device=device).long()
                    r[inds, target] = out.lb[inds, target]
                    cert_loss = loss_func(r, batch['y'])
                    loss = cur_cert_frac * cert_loss + (1.0 - cur_cert_frac) * loss
                    epoch["cert_loss"] += cert_loss.item()
                else:
                    # Bypass computing bounds during training
                    logits = out = model(batch, g, h, c, compute_bounds=False)
                    loss = loss_func(logits, batch['y'])
                epoch["total_loss"] += loss.item()
                epoch["num"] += len(root_ids)
                num_correct, num_cert_correct = num_correct_multi_classes(out[root_ids], batch['y'][root_ids],
                                                                          num_classes)
                epoch["num_correct"] += num_correct
                epoch["num_cert_correct"] += num_cert_correct
                loss.backward()
                if any(p.grad is not None and th.isnan(p.grad).any() for p in model.parameters()):
                    nan_params = [p.name for p in model.parameters() if
                                  p.grad is not None and th.isnan(p.grad).any()]
                    print('NaN found in gradients: %s' % nan_params, file=sys.stderr)
                else:
                    if args.clip_grad_norm:
                        th.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
            if args.cert_frac > 0.0:
                print(
                    "Epoch {epoch:>3}: train loss: {total_loss:.6f}, clean_loss: {clean_loss:.6f}, cert_loss: {cert_loss:.6f}".format(
                        **epoch))
            else:
                print("Epoch {epoch:>3}: train loss: {total_loss:.6f}".format(**epoch))
            sys.stdout.flush()

        epoch["clean_acc"] = 100.0 * epoch["num_correct"] / epoch["num"]
        acc_str = "  Train accuracy: {num_correct}/{num} = {clean_acc:.2f}".format(**epoch)
        if args.cert_frac > 0.0:
            epoch["cert_acc"] = 100.0 * epoch["num_cert_correct"] / epoch["num"]
            acc_str += ", certified {num_cert_correct}/{num} = {cert_acc:.2f}".format(**epoch)
        print(acc_str)
        is_best = False

        dev_results = test(args, model, devset, device, "Dev", trainset_vocab=trainset_vocab, victim_model=victim_model,
                           attack=attack)
        epoch['dev'] = dev_results
        all_epoch_stats['acc']['dev']['clean'].append(dev_results['clean_acc'])
        all_epoch_stats['acc']['dev']['cert'].append(dev_results['cert_acc'])

        dev_stats = {
            'epoch': t,
            'loss': dev_results['loss'],
            'clean_acc': dev_results['clean_acc'],
            'cert_acc': dev_results['cert_acc']
        }
        if dev_results['clean_acc'] > all_epoch_stats['acc']['best_dev']['clean'][-1]['clean_acc']:
            all_epoch_stats['acc']['best_dev']['clean'].append(dev_stats)
            if args.cert_frac == 0.0:
                is_best = True
        if dev_results['cert_acc'] > all_epoch_stats['acc']['best_dev']['cert'][-1]['cert_acc']:
            all_epoch_stats['acc']['best_dev']['cert'].append(dev_stats)
            if args.cert_frac > 0.0:
                is_best = True
        epoch['best_dev'] = {
            'clean': all_epoch_stats['acc']['best_dev']['clean'][-1],
            'cert': all_epoch_stats['acc']['best_dev']['cert'][-1]}
        all_epoch_stats["loss"]['total'].append(epoch["total_loss"])
        all_epoch_stats["loss"]['clean'].append(epoch["clean_loss"])
        all_epoch_stats["loss"]['cert'].append(epoch["cert_loss"])
        all_epoch_stats['cert']['frac'].append(epoch["cert_frac"])
        all_epoch_stats['cert']['eps'].append(epoch["cert_eps"])
        all_epoch_stats["acc"]['train']['clean'].append(epoch["clean_acc"])
        all_epoch_stats["acc"]['train']['cert'].append(epoch["cert_acc"])
        with open(os.path.join(args.out_dir, "run_stats.json"), "w") as outfile:
            json.dump(epoch, outfile)
        with open(os.path.join(args.out_dir, "all_epoch_stats.json"), "w") as outfile:
            json.dump(all_epoch_stats, outfile)
        if is_best or t == num_epochs - 1:
            if is_best:
                for fn in glob.glob(os.path.join(args.out_dir, 'model-checkpoint*.pth')):
                    os.remove(fn)
            model_save_path = os.path.join(args.out_dir, "model-checkpoint-{}.pth".format(t))
            print('Saving model to %s' % model_save_path)
            th.save(model.state_dict(), model_save_path)


def main_test(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

    testset = SSTDataset(mode='test')
    trainset_vocab = testset.vocab
    PAD_WORD = testset.PAD_WORD
    vocab, word_mat = vocabulary.Vocabulary.read_word_vecs_tree_lstm(trainset_vocab, args.glove_dir, args.glove, device)
    attack_surface = attacks.A3TWordSubstitutionAttackSurface.from_file(args.pddb_file, args.use_fewer_sub)

    testset = TextClassificationTreeDataset.from_raw_data(testset, vocab, tree_data_vocab=trainset_vocab,
                                                          PAD_WORD=PAD_WORD, attack_surface=attack_surface,
                                                          perturbation=args.perturbation)
    model = TreeLSTMDP(device,
                       len(testset.vocab),
                       word_mat.shape[1],
                       args.h_size,
                       num_classes,
                       args.dropout,
                       cell_type='childsum' if args.child_sum else 'nary',
                       pretrained_emb=word_mat, perturbation=args.perturbation,
                       no_wordvec_layer=args.no_wordvec_layer).to(device)
    if args.load_dir:
        try:
            if args.load_ckpt is None:
                load_fn = sorted(glob.glob(os.path.join(args.load_dir, 'model-checkpoint-[0-9]+.pth')))[-1]
            else:
                load_fn = os.path.join(args.load_dir, 'model-checkpoint-%d.pth' % args.load_ckpt)
            print('Loading model from %s.' % load_fn)
            state_dict = dict(th.load(load_fn, map_location=th.device('cpu')))
            model.load_state_dict(state_dict)
            print('Finished loading model.')
        except Exception as ex:
            print("Couldn't load model, starting anew: {}".format(ex))
    if args.adversary == 'exhaustive':
        adversary = ExhaustiveAdversary(attack_surface, args.perturbation)
    elif args.adversary == 'hotflip':
        victim_model = TreeModelWrapper(model, vocab, device, partial(TextClassificationTreeDataset.from_raw_data,
                                                                      tree_data_vocab=trainset_vocab,
                                                                      PAD_WORD=PAD_WORD))
        adversary = HotFlipAdversary(victim_model, args.adv_perturbation, True)
    else:
        adversary = None
    test(args, model, testset, device, "Test", adversary=adversary, trainset_vocab=trainset_vocab, PAD_WORD=PAD_WORD)


def main(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

    trainset = SSTDataset(mode='train')
    trainset_vocab = trainset.vocab
    PAD_WORD = trainset.PAD_WORD
    vocab, word_mat = vocabulary.Vocabulary.read_word_vecs_tree_lstm(trainset.vocab, args.glove_dir, args.glove, device)
    attack_surface = attacks.A3TWordSubstitutionAttackSurface.from_file(args.pddb_file, args.use_fewer_sub)
    trainset = TextClassificationTreeDataset.from_raw_data(trainset, vocab, tree_data_vocab=trainset_vocab,
                                                           PAD_WORD=PAD_WORD, attack_surface=attack_surface,
                                                           perturbation=args.perturbation)

    devset = SSTDataset(mode='dev')
    devset = TextClassificationTreeDataset.from_raw_data(devset, vocab, tree_data_vocab=trainset_vocab,
                                                         PAD_WORD=PAD_WORD, attack_surface=attack_surface,
                                                         perturbation=args.perturbation)

    testset = SSTDataset(mode='test')
    testset = TextClassificationTreeDataset.from_raw_data(testset, vocab, tree_data_vocab=trainset_vocab,
                                                          PAD_WORD=PAD_WORD, attack_surface=attack_surface,
                                                          perturbation=args.perturbation)

    model = TreeLSTMDP(device,
                       len(trainset.vocab),
                       word_mat.shape[1],
                       args.h_size,
                       num_classes,
                       args.dropout,
                       cell_type='childsum' if args.child_sum else 'nary',
                       pretrained_emb=word_mat, perturbation=args.perturbation,
                       no_wordvec_layer=args.no_wordvec_layer).to(device)
    print(model)
    if args.adv_perturbation is not None:
        victim_model = TreeModelWrapper(model, vocab, device, partial(TextClassificationTreeDataset.from_raw_data,
                                                                      tree_data_vocab=trainset_vocab,
                                                                      PAD_WORD=PAD_WORD))
        attack = HotFlipAttackTree(eval(args.adv_perturbation))
    else:
        victim_model = None
        attack = None
    train(args, model, trainset, devset, device, trainset_vocab, victim_model=victim_model, attack=attack)
    test(args, model, testset, device, "Test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', help='Directory to store and load output')
    # TreeLSTM
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--child-sum', action='store_true')
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no-wordvec-layer', action='store_true', help="Don't apply linear transform to word vectors")
    parser.add_argument('--clip-grad-norm', type=float, default=None)

    # Loading
    parser.add_argument('--load-dir', '-L', help='Where to load checkpoint')
    parser.add_argument('--load-ckpt', type=int, default=None, help='Which checkpoint to load')

    # Adversary
    parser.add_argument('--adversary', '-a', choices=['exhaustive', 'hotflip'], default=None,
                        help='Which adversary to test on')
    parser.add_argument('--use-fewer-sub', action='store_true', help='Use one substitution per word')
    parser.add_argument('--perturbation', type=str, default=None,
                        help='Perturbation for IBP training & exhaustive testing')
    parser.add_argument('--aug-perturbation', type=str, default=None, help='Perturbation for exhaustive training')
    parser.add_argument('--adv-perturbation', type=str, default=None,
                        help='Perturbation for hotflip adv training & hotflip adv testing')
    parser.add_argument('--adv-beam', type=int, default=5, help='HotFlip attack (test) beam size')
    parser.add_argument('--full-train-epochs', type=int, default=0,
                        help='If specified use full cert_frac and cert_eps for this many epochs at the end')
    parser.add_argument('--non-cert-train-epochs', type=int, default=0,
                        help='If specified train this many epochs regularly in beginning')
    parser.add_argument('--cert-frac', '-c', type=float, default=0.0,
                        help='Fraction of loss devoted to certified loss term.')

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
    parser.add_argument('--glove', '-g', choices=vocabulary.GLOVE_CONFIGS, default='840B.300d')

    # Other
    parser.add_argument('--gpu-id', type=str, default=None)
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)
    th.set_num_threads(8)
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not args.test:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        with open(os.path.join(args.out_dir, 'log.txt'), 'w') as f:
            print(sys.argv, file=f)
            print(args, file=f)
        main(args)
    else:
        main_test(args)
