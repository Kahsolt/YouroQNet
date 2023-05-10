#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/09 

# heuristically build a n-gram vocab dictionary from `train.txt`

import math
from functools import reduce
from argparse import ArgumentParser
from collections import defaultdict, Counter
from typing import Callable, Dict, Union, Tuple, List

from utils import LOG_PATH, load_dataset, timer
from mk_stats import dump_vocab, load_vocab


def make_ngram(n:int=2, line_parser:Callable=list):
  T, _ = load_dataset('train', normalize=True)
  voc = defaultdict(int)
  for t in T:
    chars = line_parser(t)
    for i in range(len(chars)-n):
      gram = ''.join(chars[i:i+n])
      voc[gram] += 1

  out_dp = LOG_PATH / f'{n}gram'
  out_dp.mkdir(exist_ok=True, parents=True)
  dump_vocab(voc, out_dp / 'vocab.txt', sort=True)


Term = Union[float, None]
Node = Tuple['Trie', Term]
Trie = Dict[str, Node]

def _vocab_to_prob(voc:Dict[str, int]) -> Dict[str, float]:
  cnt = sum(voc.values())
  for v in voc: voc[v] /= cnt
  return voc

def _mk_trie(vocab:Dict[str, float]) -> Trie:
  ''' build a trie tree '''
  def _new_node() -> Node:
    return [{}, None]
  def _add_word(trie:Trie, word:str, p:float):
    node = trie[None]
    for ch in list(word):
      if ch not in node[0]:
        node[0][ch] = _new_node()
      node = node[0][ch]
    assert node[1] is None
    node[1] = p
  
  trie = { None: _new_node() }
  for w, p in vocab.items(): _add_word(trie, w, p)
  return trie

def _q_trie(trie:Trie, sent:str) -> List[Tuple[str, float]]:
  ''' query a trie tree with given sentence, return all possible prefixing word and its prob '''
  
  node = trie[None]
  no_w = sent[:1]     # in case not present in train set
  no_p = 1e-8         # FIXME: hard-coded magic number
  NO_MATCH = (no_w, no_p)
  
  words, gram = [NO_MATCH], ''
  for ch in list(sent):
    if ch not in node[0]: break
    node = node[0][ch]
    gram += ch
    if node[1] is not None:
      words.append((gram, node[1]))
  
  return words

def _tokenize(trie:Trie, sent:str, n_beam:int=3) -> List[str]:
  candidates: List[float, str, List[str]] = [
    # Σlog(p), sent_remnant, toks
    [0.0, sent, []],
  ]

  updated = True
  while updated:
    updated = False

    candidates_new = []
    for logp, sent, toks in candidates:
      if not sent:  # just copy
        candidates_new.append([logp, sent, toks])
        continue

      init_words = _q_trie(trie, sent)
      init_words = sorted(init_words, key=lambda e: e[1], reverse=True)[:n_beam]

      for w, p in init_words:
        sent_new = sent[len(w):]
        candidates_new.append([
          logp + math.log(p),
          sent_new,
          toks + [w],
        ])
        updated = True

    candidates = sorted(candidates_new, reverse=True)[:n_beam]

  return candidates[0][-1]    # only keep the solution with highest prob

def make_tokenizer(fp:str) -> Callable[[str], List[str]]:
  trie = _mk_trie(_vocab_to_prob(load_vocab(fp)))
  return lambda t: _tokenize(trie, t, n_beam=3)

@timer
def make_kgram(vocabs:List[Dict[str, int]], min_freq:int=3, n_beam:int=3):
  # reverse list & turn freq to prob
  for i, voc in enumerate(vocabs):
    vocabs[i] = _vocab_to_prob({v: c for v, c in voc.items() if c >= min_freq})

  # merge all vocabs & make trie tree
  vocab_uni = reduce(lambda ret, voc: ret.update(voc) or ret, vocabs, {})
  trie = _mk_trie(vocab_uni)

  # tokenize T with trie
  T, _ = load_dataset('train', normalize=True)
  T_toks: List[str] = reduce(lambda ret, sent: ret.extend(_tokenize(trie, sent, n_beam)) or ret, T, [])

  # collect all tokens as the new vocab
  out_dp = LOG_PATH / 'kgram'
  out_dp.mkdir(exist_ok=True, parents=True)
  dump_vocab(Counter(T_toks), out_dp / 'vocab.txt', sort=True)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--ngram', action='store_true', help='make ngram vocab')
  parser.add_argument('--kgram', action='store_true', help='make kgram vocab')
  parser.add_argument('--min_freq', default=3, type=int, help='kgram min_freq')
  parser.add_argument('--n_beam',   default=3, type=int, help='kgram n_beam')
  args = parser.parse_args()

  Ns = [2, 3, 4, 5]

  if args.ngram:
    # fixed n-gram
    for n in Ns:
      print(f'>> making {n}gram ...')
      make_ngram(n)

  if args.kgram:
    # adaptive k-gram
    vocabs = []
    for n in Ns:
      vocabs.append(load_vocab(LOG_PATH / f'{n}gram' / 'vocab.txt'))
    print('>> making kgram ...')
    make_kgram(vocabs, min_freq=args.min_freq, n_beam=args.n_beam)
