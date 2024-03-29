# Certified Robustness to Programmable Transformations in LSTMs

This is the official GitHub repository for the following paper:

Certified Robustness to Programmable Transformations in LSTMs

## Setup

### Download Dataset

A large portion of this repo comes from https://github.com/robinjia/certified-word-sub. Please use `./download_deps.sh` to download IMDB dataset, Glove embeddings, LM scores, and Counterfitted vectors. The test set of SST2 can be found [here](https://github.com/ForeverZyh/A3T/blob/master/dataset/sst2test.txt). Other part of SST2 and SST will be download automatically by the TensorFlow Dataset and DGL.

For the comparison between SAFER, please use https://github.com/lushleaf/Structure-free-certified-NLP to generate the synonym set `data/imdb_neighbor_constraint_pca0.8.pkl` and `data/imdb_perturbation_constraint_pca0.8_100.pkl`.

For the comparison between POPQORN, please use the code under https://github.com/ForeverZyh/POPQORN and put the LSTM model under `NewsTitleClassification`.

For the comparison between ASCC, please use the code under https://github.com/ForeverZyh/ASCC. The code will generate LSTM model `bilstm_adv_best_pth` and numpy files `index_word.npy` and `word_index.npy`, which can be directly loaded by `train.py` in this repository. 

### Environment

We use conda to setup the virtual environment for Python version and CUDA version. We use Python3.6, and let conda resolve CUDA version conflicts for us. Please install the following packages:

```
tensorflow==1.13.1
keras==2.3.1
tensorflow-datasets==1.3.2
nltk==3.4
torch==1.5.1
dgl==0.5.2
```

We also provide the environment of conda packages in `packages.txt`.

## Experiment

We provide all training scripts in `scripts.md`. Please use `src/train.py` to train and test LSTM and Bi-LSTM and `src/train_tree_lstm.py` to train Tree-LSTM.
