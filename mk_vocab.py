#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/09 

# heuristically build a n-gram vocab dictionary from `train.txt`

from collections import defaultdict
from typing import Callable, Dict

from utils import *
from mk_stats import dump_vocab


def make_ngram(n:int=2, line_parser:Callable=list):
  out_dp = LOG_PATH / f'{n}gram'
  out_dp.mkdir(exist_ok=True, parents=True)

  T, _ = load_dataset('train')
  voc = defaultdict(int)
  for t in T:
    chars = line_parser(t)
    for i in range(len(chars)-n):
      gram = ''.join(chars[i:i+n])
      voc[gram] += 1

  dump_vocab(voc, out_dp / 'vocab.txt', sort=True)
  return voc


def make_kgram(vocabs:List[Dict[str, int]]):
  out_dp = LOG_PATH / 'kgram'
  out_dp.mkdir(exist_ok=True, parents=True)

  T, _ = load_dataset('train')
  voc = defaultdict(int)



  dump_vocab(voc, out_dp / 'vocab.txt', sort=True)


if __name__ == '__main__':
  # fixed n-gram
  vocabs = []
  for n in [2, 3, 4, 5]:
    print(f'making vocab for {n}gram ...')
    vocabs.append(make_ngram(n))

  # adaptive k-gram
  print('making vocab for kgram ...')
  make_kgram(vocabs)
