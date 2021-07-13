# Scripts

## LSTM on SST2

### Normal Training

```bash
# Train
python src/train.py classification lstm-dp outdir_lstm_cert_sst2 --pool final --no-bidirectional -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5

# Exhaustive [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2 --load-ckpt 4 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2 --load-ckpt 4 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only

# Exhaustive [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2 --load-ckpt 4 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2 --load-ckpt 4 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only

# Exhaustive [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2 --load-ckpt 4 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only -D --perturbation "[(Del,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2 --load-ckpt 4 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only

# Exhaustive S_{Review}
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2 --load-ckpt 4 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only -D --perturbation "[(Trans1, 1), (Trans2, 1), (Trans3, 1), (Trans4, 1), (Sub, 2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip S_{Review}
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2 --load-ckpt 4 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Trans1(), 1), (Trans2(), 1), (Trans3(), 1), (Trans4(), 1), (Sub('data/pddb', True),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only
```



### ARC

```bash
# Train [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp outdir_lstm_cert_sst2_sub2del2_fewer_0.5 --pool final --no-bidirectional -d 100 -T 24 --full-train-epochs 8 -c 0.5 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub

# Exhaustive [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2del2_fewer_0.5 --load-ckpt 22 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2del2_fewer_0.5 --load-ckpt 22 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Train [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_lstm_cert_sst2_sub2ins2_fewer_0.6 --pool final --no-bidirectional -d 100 -T 24 --full-train-epochs 8 -c 0.6 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub

# Exhaustive [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2ins2_fewer_0.6 --load-ckpt 22 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a exhaustive --adv-only

# HotFlip [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2ins2_fewer_0.6 --load-ckpt 22 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only

# Train [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_lstm_cert_sst2_del2ins2_k0.9 --pool final --no-bidirectional -d 100 -T 24 --full-train-epochs 8 -c 0.9 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Del,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d"

# Exhaustive [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_del2ins2_k0.9 --load-ckpt 17 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Del,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test -a exhaustive --adv-only

# HotFlip [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_del2ins2_k0.9 --load-ckpt 17 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test -a hotflip --adv-only

# Train S_{Review}
python src/train.py classification lstm-dp-general outdir_vivid --pool final --no-bidirectional -d 100 -T 24 --full-train-epochs 8 -c 0.8 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Trans1, 1), (Trans2, 1), (Trans3, 1), (Trans4, 1), (Sub, 2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d"

# Exhaustive S_{Review}
python src/train.py classification lstm-dp-general out -L outdir_vivid --load-ckpt 19 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Trans1, 1), (Trans2, 1), (Trans3, 1), (Trans4, 1), (Sub, 2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test -a exhaustive --adv-only

# HotFlip S_{Review}
python src/train.py classification lstm-dp-general out -L outdir_vivid --load-ckpt 19 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Trans1(), 1), (Trans2(), 1), (Trans3(), 1), (Trans4(), 1), (Sub('data/pddb', True),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test -a hotflip --adv-only
```



### HotFlip

```bash
# Train [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp outdir_lstm_cert_sst2_sub2del2_fewer_hotflip --pool final --no-bidirectional -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5

# Exhaustive [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2del2_fewer_hotflip --load-ckpt 6 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2del2_fewer_hotflip --load-ckpt 6 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Train [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_lstm_cert_sst2_sub2ins2_fewer_hotflip --pool final --no-bidirectional -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5

# Exhaustive [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2ins2_fewer_hotflip --load-ckpt 5 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2ins2_fewer_hotflip --load-ckpt 5 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only

# Train [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_lstm_cert_sst2_del2ins2_fewer_hotflip --pool final --no-bidirectional -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5

# Exhaustive [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_del2ins2_fewer_hotflip --load-ckpt 5 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only -D --perturbation "[(Del,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_del2ins2_fewer_hotflip --load-ckpt 5 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only

# Train S_{Review}
python src/train.py classification lstm-dp outdir_vivid_hotflip --pool final --no-bidirectional -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Trans1(), 1), (Trans2(), 1), (Trans3(), 1), (Trans4(), 1), (Sub('data/pddb', True),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5

# Exhaustive S_{Review}
python src/train.py classification lstm-dp-general out -L outdir_vivid_hotflip --load-ckpt 8 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only -D --perturbation "[(Trans1, 1), (Trans2, 1), (Trans3, 1), (Trans4, 1), (Sub, 2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip S_{Review}
python src/train.py classification lstm-dp-general out -L outdir_vivid_hotflip --load-ckpt 8 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Trans1(), 1), (Trans2(), 1), (Trans3(), 1), (Trans4(), 1), (Sub('data/pddb', True),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only
```



###Data Aug

```bash
# Train [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp outdir_lstm_cert_sst2_sub2del2_fewer_dataaug --pool final --no-bidirectional -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-random-aug --early-stopping 5

# Exhaustive [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2del2_fewer_dataaug --load-ckpt 4 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2del2_fewer_dataaug --load-ckpt 4 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Train [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_lstm_cert_sst2_sub2ins2_fewer_dataaug --pool final --no-bidirectional -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-random-aug --early-stopping 5

# Exhaustive [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2ins2_fewer_dataaug --load-ckpt 10 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_sub2ins2_fewer_dataaug --load-ckpt 10 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only

# Train [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_lstm_cert_sst2_del2ins2_fewer_dataaug --pool final --no-bidirectional -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5 --use-random-aug

# Exhaustive [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_del2ins2_fewer_dataaug --load-ckpt 11 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Del,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_del2ins2_fewer_dataaug --load-ckpt 11 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only

# Train S_{Review}
python src/train.py classification lstm-dp outdir_vivid_dataaug --pool final --no-bidirectional -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Trans1(), 1), (Trans2(), 1), (Trans3(), 1), (Trans4(), 1), (Sub('data/pddb', True),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5 --use-random-aug

# Exhaustive S_{Review}
python src/train.py classification lstm-dp-general out -L outdir_vivid_dataaug --load-ckpt 7 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Trans1, 1), (Trans2, 1), (Trans3, 1), (Trans4, 1), (Sub, 2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip S_{Review}
python src/train.py classification lstm-dp-general out -L outdir_vivid_dataaug --load-ckpt 7 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]" --adv-perturbation="[(Trans1(), 1), (Trans2(), 1), (Trans3(), 1), (Trans4(), 1), (Sub('data/pddb', True),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --test --use-fewer-sub -a hotflip --adv-only
```



### Tree-LSTM on SST

### Normal Training

```bash
# Train
python src/train_tree_lstm.py tree --perturbation "[(Sub,1)]" --epoch 50 -c 0 --use-fewer-sub --clip-grad-norm=0.25 --early-stopping 10

# Exhaustive [(Sub,2), (Del,2)]
python src/train_tree_lstm.py out -L tree --load-ckpt 22 --test -a exhaustive --perturbation "[(Sub,2), (Del,2)]" --use-fewer-sub

# HotFlip [(Sub,2), (Del,2)]
python src/train_tree_lstm.py out -L tree --load-ckpt 22 --test -a hotflip --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub

# Exhaustive [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree --load-ckpt 22 --test -a exhaustive --perturbation "[(Sub,2), (Ins,2)]" --use-fewer-sub

# HotFlip [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree --load-ckpt 22 --test -a hotflip --adv-perturbation "[(Sub('data/pddb', True),2), (Ins(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub

# Exhaustive [(Del,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree --load-ckpt 22 --test -a exhaustive --perturbation "[(Del,2), (Ins,2)]" --use-fewer-sub

# HotFlip [(Del,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree --load-ckpt 22 --test -a hotflip --adv-perturbation "[(Del(), 2), (Ins(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub
```



### ARC

```bash
# Train [(Sub,2), (Del,2)]
python src/train_tree_lstm.py tree_del2sub2_k0.07 --perturbation "[(Sub,2),(Del,2)]" -c 0.07 --use-fewer-sub --epoch 30 --full-train-epochs=10 --clip-grad-norm=0.25

# Exhaustive [(Sub,2), (Del,2)]
python src/train_tree_lstm.py out -L tree_del2sub2_k0.07 --load-ckpt 26 --test -a exhaustive --perturbation "[(Sub,2), (Del,2)]" --use-fewer-sub

# HotFlip [(Sub,2), (Del,2)]
python src/train_tree_lstm.py out -L tree_del2sub2_k0.07 --load-ckpt 26 --test -a hotflip --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub

# Train [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py tree_ins2sub2_k0.06 --perturbation "[(Ins,2),(Sub,2)]" -c 0.06 --use-fewer-sub --epoch 30 --full-train-epochs=10 --clip-grad-norm=0.25

# Exhaustive [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_ins2sub2_k0.06 --load-ckpt 27 --test -a exhaustive --perturbation "[(Sub,2),(Ins,2)]" --use-fewer-sub

# HotFlip [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_ins2sub2_k0.06 --load-ckpt 27 --test -a hotflip --adv-perturbation "[(Sub('data/pddb', True),2), (Ins(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub

# Train [(Del,2), (Ins,2)]
python src/train_tree_lstm.py tree_del2ins2_k0.07 --perturbation "[(Ins,2),(Del,2)]" -c 0.07 --use-fewer-sub --epoch 30 --full-train-epochs=10 --clip-grad-norm=0.25

# Exhaustive [(Del,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_del2ins2_k0.07 --load-ckpt 26 --test -a exhaustive --perturbation "[(Ins,2),(Del,2)]" --use-fewer-sub

# HotFlip [(Del,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_del2ins2_k0.07 --load-ckpt 26 --test -a hotflip --adv-perturbation "[(Del(),2), (Ins(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub
```



### HotFlip

```bash
# Train [(Sub,2), (Del,2)]
python src/train_tree_lstm.py tree_del2sub2_hotflip --perturbation "[(Sub,1)]" --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" -c 0 --use-fewer-sub --epoch 30 --clip-grad-norm=0.25 --early-stopping 5

# Exhaustive [(Sub,2), (Del,2)]
python src/train_tree_lstm.py out -L tree_del2sub2_hotflip --load-ckpt 16 --test -a exhaustive --perturbation "[(Sub,2), (Del,2)]" --use-fewer-sub

# HotFlip [(Sub,2), (Del,2)]
python src/train_tree_lstm.py out -L tree_del2sub2_hotflip --load-ckpt 16 --test -a hotflip --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub

# Train [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py tree_ins2sub2_hotflip --perturbation "[(Sub,1)]" --adv-perturbation "[(Sub('data/pddb', True),2), (Ins(),2)]" -c 0 --use-fewer-sub --epoch 30 --clip-grad-norm=0.25 --early-stopping 5

# Exhaustive [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_ins2sub2_hotflip --load-ckpt 4 --test -a exhaustive --perturbation "[(Sub,2), (Ins,2)]" --use-fewer-sub

# HotFlip [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_ins2sub2_hotflip --load-ckpt 4 --test -a hotflip --adv-perturbation "[(Sub('data/pddb', True),2), (Ins(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub

# Train [(Del,2), (Ins,2)]
python src/train_tree_lstm.py tree_del2ins2_hotflip --perturbation "[(Sub,1)]" --adv-perturbation "[(Del(),2), (Ins(),2)]" -c 0 --use-fewer-sub --epoch 30 --clip-grad-norm=0.25 --early-stopping 5

# Exhaustive [(Del,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_del2ins2_hotflip --load-ckpt 12 --test -a exhaustive --perturbation "[(Del,2), (Ins,2)]" --use-fewer-sub

# HotFlip [(Del,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_del2ins2_hotflip --load-ckpt 12 --test -a hotflip --adv-perturbation "[(Del(),2), (Ins(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub
```



### Data Aug

```bash
# Train [(Sub,2), (Del,2)]
python src/train_tree_lstm.py tree_del2sub2_dataaug --perturbation "[(Sub,1)]" --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" -c 0 --use-fewer-sub --epoch 30 --clip-grad-norm=0.25 --early-stopping 5 --use-random-aug

# Exhaustive [(Sub,2), (Del,2)]
python src/train_tree_lstm.py out -L tree_del2sub2_dataaug --load-ckpt 7 --test -a exhaustive --perturbation "[(Sub,2), (Del,2)]" --use-fewer-sub

# HotFlip [(Sub,2), (Del,2)]
python src/train_tree_lstm.py out -L tree_del2sub2_dataaug --load-ckpt 7 --test -a hotflip --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub

# Train [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py tree_ins2sub2_dataaug --perturbation "[(Sub,1)]" --adv-perturbation "[(Sub('data/pddb', True),2), (Ins(),2)]" -c 0 --use-fewer-sub --epoch 30 --clip-grad-norm=0.25 --early-stopping 5 --use-random-aug

# Exhaustive [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_ins2sub2_dataaug --load-ckpt 14 --test -a exhaustive --perturbation "[(Sub,2), (Ins,2)]" --use-fewer-sub

# HotFlip [(Sub,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_ins2sub2_dataaug --load-ckpt 14 --test -a hotflip --adv-perturbation "[(Sub('data/pddb', True),2), (Ins(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub

# Train [(Del,2), (Ins,2)]
python src/train_tree_lstm.py tree_del2ins2_dataaug --perturbation "[(Sub,1)]" --adv-perturbation "[(Del(),2), (Ins(),2)]" -c 0 --use-fewer-sub --epoch 30 --clip-grad-norm=0.25 --early-stopping 5 --use-random-aug

# Exhaustive [(Del,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_del2ins2_dataaug --load-ckpt 7 --test -a exhaustive --perturbation "[(Del,2), (Ins,2)]" --use-fewer-sub

# HotFlip [(Del,2), (Ins,2)]
python src/train_tree_lstm.py out -L tree_del2ins2_dataaug --load-ckpt 7 --test -a hotflip --adv-perturbation "[(Del(),2), (Ins(),2)]" --perturbation "[(Sub,1)]" --use-fewer-sub
```



## Bi-LSTM on SST2

### Normal Training

```bash
# Train
python src/train.py classification lstm-dp outdir_lstm_cert_sst2_bi --pool final -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5

# Exhaustive [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_bi --load-ckpt 3 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_bi --load-ckpt 3 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Exhaustive [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_bi --load-ckpt 3 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_bi --load-ckpt 3 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Exhaustive [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_bi --load-ckpt 3 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Del,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_sst2_bi --load-ckpt 3 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only
```



### ARC

```bash
# Train [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp outdir_bilstm_cert_sst2_sub2del2_fewer_0.5 --pool final -d 100 -T 24 --full-train-epochs 8 -c 0.5 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub

# Exhaustive [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2del2_fewer_0.5 --load-ckpt 16 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2del2_fewer_0.5 --load-ckpt 16 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Train [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_bilstm_cert_sst2_sub2ins2_fewer_0.9 --pool final -d 100 -T 24 --full-train-epochs 8 -c 0.9 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub

# Exhaustive [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2ins2_fewer_0.9 --load-ckpt 21 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2ins2_fewer_0.9 --load-ckpt 21 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Train [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_bilstm_cert_sst2_del2ins2_fewer_0.7 --pool final -d 100 -T 24 --full-train-epochs 8 -c 0.7 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Del,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub

# Exhaustive [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_del2ins2_fewer_0.7 --load-ckpt 22 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Del,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L  outdir_bilstm_cert_sst2_del2ins2_fewer_0.7 --load-ckpt 22 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only
```



### HotFlip

```bash
# Train [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp outdir_bilstm_cert_sst2_sub2del2_fewer_hotflip --pool final -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5

# Exhaustive [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2del2_fewer_hotflip --load-ckpt 4 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2del2_fewer_hotflip --load-ckpt 4 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Train [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_bilstm_cert_sst2_sub2ins2_fewer_hotflip --pool final -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5

# Exhaustive [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2ins2_fewer_hotflip --load-ckpt 19 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2ins2_fewer_hotflip --load-ckpt 19 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Train [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_bilstm_cert_sst2_del2ins2_fewer_hotflip --pool final -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5

# Exhaustive [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_del2ins2_fewer_hotflip --load-ckpt 4 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Del,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_del2ins2_fewer_hotflip --load-ckpt 4 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only
```



### Data Aug

```bash
# Train [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp outdir_bilstm_cert_sst2_sub2del2_fewer_dataaug --pool final -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5 --use-random-aug

# Exhaustive [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2del2_fewer_dataaug --load-ckpt 2 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Del,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2del2_fewer_dataaug --load-ckpt 2 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Del(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Train [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_bilstm_cert_sst2_sub2ins2_fewer_dataaug --pool final -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5 --use-random-aug

# Exhaustive [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2ins2_fewer_dataaug --load-ckpt 8 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_sub2ins2_fewer_dataaug --load-ckpt 8 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only

# Train [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp outdir_bilstm_cert_sst2_del2ins2_fewer_dataaug --pool final -d 100 -T 24 -c 0 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation="[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --early-stopping 5 --use-random-aug

# Exhaustive [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_del2ins2_fewer_dataaug --load-ckpt 8 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Del,2), (Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Del,2), (Ins,2)]
python src/train.py classification lstm-dp out -L outdir_bilstm_cert_sst2_del2ins2_fewer_dataaug --load-ckpt 8 --pool final -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Del(),2), (Ins(),2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only
```



## ARC Compare to A3T

The training scripts and trained models are the same as (Sec. LSTM on SST2)

#### ARC + A3T

```bash
# Train [(Sub, 2), (Del, 2)], abstract Sub
python src/train.py classification lstm-dp sst2_sub2del2_A3T_abs_sub --pool final --no-bidirectional -d 100 -T 24 --full-train-epochs 8 -c 0.8 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,2)]" --aug-perturbation "[(Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --eaugment-by 2

# Test [(Sub, x), (Del, x)] (x=1..3)
python src/train.py classification lstm-dp out -L sst2_sub2del2_A3T_abs_sub --load-ckpt 22 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,x), (Del,x)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# Train [(Sub, 2), (Del, 2)], abstract Del
python src/train.py classification lstm-dp sst2_sub2del2_A3T_abs_del --pool final --no-bidirectional -d 100 -T 24 --full-train-epochs 8 -c 0.8 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Del,2)]" --aug-perturbation "[(Sub,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --eaugment-by 2

# Test [(Sub, x), (Del, x)] (x=1..3)
python src/train.py classification lstm-dp out -L sst2_sub2del2_A3T_abs_del --load-ckpt 23 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,x), (Del,x)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# Train [(Sub, 2), (Ins, 2)], abstract Sub
python src/train.py classification lstm-dp sst2_sub2ins2_A3T_abs_sub --pool final --no-bidirectional -d 100 -T 24 --full-train-epochs 8 -c 0.8 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,2)]" --aug-perturbation "[(Ins,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --eaugment-by 2

# Test [(Sub, x), (Ins, x)] (x=1..3)
python src/train.py classification lstm-dp out -L sst2_sub2ins2_A3T_abs_sub --load-ckpt 18 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,x), (Ins,x)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# Train [(Sub, 2), (Ins, 2)], abstract Ins
python src/train.py classification lstm-dp sst2_sub2ins2_A3T_abs_ins --pool final --no-bidirectional -d 100 -T 24 --full-train-epochs 8 -c 0.8 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Ins,2)]" --aug-perturbation "[(Sub,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --eaugment-by 2

# Test [(Sub, x), (Ins, x)] (x=1..3)
python src/train.py classification lstm-dp out -L sst2_sub2ins2_A3T_abs_ins --load-ckpt 22 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,x), (Ins,x)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only
```



## ARC Compare to CertSub

```bash
# Train [(Sub, 1)]
python src/train.py classification lstm-dp outdir_lstm_cert_dp_1 -d 100 --pool mean -T 30 --dropout-prob 0.2 --full-train-epochs 10 -c 0.8 -b 32 --save-best-only --dataset Imdb --perturbation "[(Sub, 1)]"

# Exhaustive [(Sub, 1)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_dp_1 --load-ckpt 22 -d 100 --pool mean -T 0 --test -b 1 --perturbation="[(Sub,1)]" --use-lm --dataset "Imdb" --downsample-to 1000 --downsample-shard 0 -a exhaustive --adv-only

# Train [(Sub, 2)]
python src/train.py classification lstm-dp outdir_lstm_cert_dp_2 -d 100 --pool mean -T 30 --dropout-prob 0.2 --full-train-epochs 10 -c 0.8 -b 32 --save-best-only --dataset Imdb --perturbation "[(Sub, 2)]"

# Exhaustive [(Sub, 2)]
python src/train.py classification lstm-dp out -L outdir_lstm_cert_dp_2 --load-ckpt 28 -d 100 --pool mean -T 0 --test -b 1 --perturbation="[(Sub,2)]" --use-lm --dataset "Imdb" --downsample-to 1000 --downsample-shard 0 -a exhaustive --adv-only
```



## ARC Compare to ASCC

For training the ASCC model, please see https://github.com/ForeverZyh/ASCC.

```bash
# Exhaustive [(Sub, 1)]
python src/train.py classification lstm-dp-ascc out -L ascc -d 100 --pool mean -T 0 --test -b 1 --perturbation="[(Sub,1)]" --use-lm --dataset "Imdb" --downsample-to 1000 --downsample-shard 0 -a exhaustive --adv-only

# Exhaustive [(Sub, 2)]
python src/train.py classification lstm-dp-ascc out -L ascc -d 100 --pool mean -T 0 --test -b 1 --perturbation="[(Sub,2)]" --use-lm --dataset "Imdb" --downsample-to 1000 --downsample-shard 0 -a exhaustive --adv-only
```



## ARC Compare to POPQORN & Huang et al.

Training the `[(Sub, 3)]` model:

```bash
# Train [(Sub,3)]
python src/train.py classification lstm-dp sst2_sub3 --pool final --no-bidirectional -d 100 -T 24 --full-train-epochs 8 -c 0.6 --dropout-prob 0.2 -b 32 --save-best-only --perturbation "[(Sub,2), (Del,2)]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub

# Exhaustive [(Sub,3)]
python src/train.py classification lstm-dp out -L sst2_sub3 --load-ckpt 23 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,3]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a exhaustive --adv-only

# HotFlip [(Sub,3)]
python src/train.py classification lstm-dp out -L sst2_sub3 --load-ckpt 23 --pool final --no-bidirectional -d 100 -T 0 -b 1 --save-best-only --perturbation "[(Sub,1)]"  --adv-perturbation "[(Sub('data/pddb', True),3]" --dataset "SST2" --use-a3t-settings -g "6B.300d" --use-fewer-sub --test -a hotflip --adv-only
```

For POPQORN testing, please see https://github.com/ForeverZyh/POPQORN.



## ARC Evalaute on SAFER

```bash
# Train [(Sub, 1)] 
python src/train.py classification lstm-dp outdir_RS_dp_sub1 -d 100 --pool mean -T 30 --dropout-prob 0.2 --full-train-epochs 10 -c 0.8 -b 32 --save-best-only --dataset Imdb --use-RS-settings --perturbation "[(Sub, 1)]"

# Randomized Smoothing [(Sub, 1)]
python src/train.py classification lstm-dp out -L outdir_RS_dp_sub1 --load-ckpt 24 -d 100 --pool mean -T 0 --test -b 1 --perturbation="[(Sub,1)]" --use-RS-settings --dataset "Imdb" --downsample-to 1000 --downsample-shard 0 -a RS --adv-only  

# Exhaustive [(Sub, 1)]
python src/train.py classification lstm-dp out -L outdir_RS_dp_sub1 --load-ckpt 24 -d 100 --pool mean -T 0 --test -b 1 --perturbation="[(Sub,1)]" --use-RS-settings --dataset "Imdb" --downsample-to 1000 --downsample-shard 0 -a exhaustive --adv-only

# Train [(Sub, 2)]
python src/train.py classification lstm-dp outdir_RS_dp_sub2 -d 100 --pool mean -T 30 --dropout-prob 0.2 --full-train-epochs 10 -c 0.8 -b 32 --save-best-only --dataset Imdb --use-RS-settings --perturbation "[(Sub,2)]" 

# Randomized Smoothing [(Sub, 2)]
python src/train.py classification lstm-dp out -L outdir_RS_dp_sub2 --load-ckpt 23 -d 100 --pool mean -T 0 --test -b 1 --perturbation="[(Sub,2)]" --use-RS-settings --dataset "Imdb" --downsample-to 1000 --downsample-shard 0 -a RS --adv-only

# Exhaustive [(Sub, 2)]
python src/train.py classification lstm-dp out -L outdir_RS_dp_sub2 --load-ckpt 23 -d 100 --pool mean -T 0 --test -b 1 --perturbation="[(Sub,2)]" --use-RS-settings --dataset "Imdb" --downsample-to 1000 --downsample-shard 0 -a exhaustive --adv-only
```

