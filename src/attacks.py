"""Defines an attack surface."""
import collections
import json
import sys
import os
import numpy as np

from nltk import pos_tag

OPTS = None

DEFAULT_MAX_LOG_P_DIFF = -5.0  # Maximum difference in log p for swaps.

class AttackSurface(object):
  def get_swaps(self, words):
    """Return valid substitutions for each position in input |words|."""
    raise NotImplementedError

class WordSubstitutionAttackSurface(AttackSurface):
  def __init__(self, neighbors):
    self.neighbors = neighbors

  @classmethod
  def from_file(cls, neighbors_file):
    with open(neighbors_file) as f:
      return cls(json.load(f))

  def get_swaps(self, words):
    swaps = []
    for i in range(len(words)):
      if words[i] in self.neighbors:
        swaps.append(self.neighbors[words[i]])
      else:
        swaps.append([])
    return swaps

class A3TWordSubstitutionAttackSurface(AttackSurface):

  def __init__(self, synonym_dict, synonym_dict_pos_tag):
    self.synonym_dict = synonym_dict
    self.synonym_dict_pos_tag = synonym_dict_pos_tag

  @classmethod
  def from_file(cls, pddb_path):
    try:
      synonym_dict = dict(
        np.load(os.path.join(pddb_path, 'synonym_dict.npy'), allow_pickle=True).item())
      synonym_dict_pos_tag = dict(
        np.load(os.path.join(pddb_path, 'synonym_dict_pos_tag.npy'), allow_pickle=True).item())
      print("Loading cached synonym_dict success!")
      return cls(synonym_dict, synonym_dict_pos_tag)
    except:
      pass

    synonym_dict = {}
    synonym_dict_pos_tag = {}
    pddb_files = [f for f in os.listdir(pddb_path) if os.path.isfile(os.path.join(pddb_path, f)) and f[:4] == "ppdb"]
    if len(pddb_files) == 0:
      raise AttributeError("No PPDB files found in %s" % pddb_path)
    else:
      pddb_file = pddb_files[0]
      print("Using ", pddb_files, " ...")

    lines = open(os.path.join(pddb_path, pddb_file)).readlines()
    for line in lines:
      tmp = line.strip().split(" ||| ")
      pos_tag, x, y = tmp[0][1:-1], tmp[1], tmp[2]
      A3TWordSubstitutionAttackSurface.synonym_dict_add_str(synonym_dict, synonym_dict_pos_tag, x, y, pos_tag)
      A3TWordSubstitutionAttackSurface.synonym_dict_add_str(synonym_dict, synonym_dict_pos_tag, y, x, pos_tag)

    np.save(os.path.join(pddb_path, 'synonym_dict.npy'), synonym_dict)
    np.save(os.path.join(pddb_path, 'synonym_dict_pos_tag.npy'), synonym_dict_pos_tag)
    print("Loading synonym_dict success!")
    return cls(synonym_dict, synonym_dict_pos_tag)

  @staticmethod
  def synonym_dict_add_str(synonym_dict, synonym_dict_pos_tag, x, y, pos_tag):
    if x not in synonym_dict:
      synonym_dict[x] = [y]
      synonym_dict_pos_tag[x] = [pos_tag]
    else:
      synonym_dict[x].append(y)
      synonym_dict_pos_tag[x].append(pos_tag)

  def get_swaps(self, words):
    pos_tags = pos_tag(words)
    swaps = []
    for i in range(len(words)):
      if words[i] in self.synonym_dict:
        synonyms = self.synonym_dict[words[i]]
        synonyms_pos_tag = self.synonym_dict_pos_tag[words[i]]
        synonyms = list(set([x for (x, y) in zip(synonyms, synonyms_pos_tag) if y == pos_tags[i][1]]))
        swaps.append(synonyms)
      else:
        swaps.append([])
    return swaps

class LMConstrainedAttackSurface(AttackSurface):
  """WordSubstitutionAttackSurface with language model constraint."""
  def __init__(self, neighbors, lm_scores, min_log_p_diff=DEFAULT_MAX_LOG_P_DIFF):
    self.neighbors = neighbors
    self.lm_scores = lm_scores
    self.min_log_p_diff = min_log_p_diff

  @classmethod
  def from_files(cls, neighbors_file, lm_file):
    with open(neighbors_file) as f:
      neighbors = json.load(f)
    with open(lm_file) as f:
      lm_scores = {}
      cur_sent = None
      for line in f:
        toks = line.strip().split('\t')
        if len(toks) == 2:
          cur_sent = toks[1].lower()
          lm_scores[cur_sent] = collections.defaultdict(dict)
        else:
          word_idx, word, score = int(toks[1]), toks[2], float(toks[3])
          lm_scores[cur_sent][word_idx][word] = score
    return cls(neighbors, lm_scores)

  def get_swaps(self, words):
    swaps = []
    words = [word.lower() for word in words]
    s = ' '.join(words)
    if s not in self.lm_scores:
      raise KeyError('Unrecognized sentence "%s"' % s)
    for i in range(len(words)):
      if i in self.lm_scores[s]:
        cur_swaps = []
        orig_score = self.lm_scores[s][i][words[i]]
        for swap, score in self.lm_scores[s][i].items():
          if swap == words[i]: continue
          if swap not in self.neighbors[words[i]]: continue
          if score - orig_score >= self.min_log_p_diff:
            cur_swaps.append(swap)
        swaps.append(cur_swaps)
      else:
        swaps.append([])
    return swaps
