#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/03 

from pathlib import Path
from functools import reduce
from copy import deepcopy
from collections import Counter, defaultdict, OrderedDict
from typing import List, Dict, Callable, Any

import jieba
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path('data')
STAT_PATH = Path('stat') ; STAT_PATH.mkdir(exist_ok=True)

LIMITS  = [0, 4, 12, 32]
N_CLASS = 4


def load_vocab(fp:str) -> Dict[str, int]:
  with open(fp, encoding='utf-8') as fh:
    lines = fh.read().strip().split('\n')
  val_cnt = [line.split('\t') for line in lines]
  return {v: int(c) for v, c in val_cnt}


def dump_vocab(voc:Dict[str, int], fp:str):
  with open(fp, 'w', encoding='utf-8') as fh:
    for v, c in voc.items():
      fh.write(f'{v}\t{c}\n')


def write_stats(items:List[Any], limit:int, name:str, subfolder:str=''):
  out_dp = STAT_PATH / subfolder
  out_dp.mkdir(exist_ok=True)

  pairs = sorted([(c, v) for v, c in Counter(items).items() if c > limit], reverse=True)
  dump_vocab(OrderedDict([(v, c) for c, v in pairs]), out_dp / f'vocab_{name}.txt')

  with open(out_dp / f'stats_{name}.txt', 'w', encoding='utf-8') as fh:
    fh.write(f'words: {len(items)}\n')
    fh.write(f'vocab: {len(pairs)}\n')

  plt.clf()
  plt.plot([c for c, v in pairs])
  plt.suptitle(f'freq_{name}')
  plt.savefig(out_dp / f'freq_{name}.png')


def make_stats(kind:str, limit:int, line_parser:Callable):
  subfolder = kind if limit <= 0 else f'{kind}-{limit}'

  words_all = []
  for split in ['train', 'test']:
    df = pd.read_csv(DATA_PATH / f'{split}.csv')
    label, text = df['label'].to_numpy().astype(int), df['text'].to_numpy()

    words_cls = defaultdict(list)
    for cls, line in zip(label, text):
      words_cls[cls].extend(line_parser(line))

    # class-wise
    for cls in words_cls.keys():
      write_stats(words_cls[cls], limit, f'{split}_{cls}', subfolder)
    # split-wise
    words_split = reduce(lambda x, y: x.extend(y) or x, words_cls.values(), [])
    write_stats(words_split, limit, split, subfolder)

    words_all.extend(words_split)

  # dataset-wise
  write_stats(words_all, limit, 'all', subfolder)


def diff_vocab(kind:str, limit:int, splits:List[str]):
  subfolder = kind if limit <= 0 else f'{kind}-{limit}'
  out_dp = STAT_PATH / subfolder

  vocabs = { split: load_vocab(out_dp / f'vocab_{split}.txt') for split in splits }

  for split1 in splits:
    for split2 in splits:
      if split1 == split2: continue

      voc1 = deepcopy(vocabs[split1])
      voc2 = vocabs[split2]
      for key in voc2:
        if key in voc1:
          del voc1[key]

      dump_vocab(voc1, out_dp / f'vocab_{split1}-{split2}.txt')


def uniq_vocab(kind:str, limit:int, splits:List[str]):
  subfolder = kind if limit <= 0 else f'{kind}-{limit}'
  out_dp = STAT_PATH / subfolder

  for split in splits:
    vocabs = { cls: load_vocab(out_dp / f'vocab_{split}_{cls}.txt') for cls in range(N_CLASS) }

    for cls1 in range(N_CLASS):
      voc1 = deepcopy(vocabs[cls1])
      for cls2 in range(N_CLASS):
        if cls1 == cls2: continue

        voc2 = vocabs[cls2]
        for key in voc2:
          if key in voc1:
            del voc1[key]
      
      dump_vocab(voc1, out_dp / f'vocab_{split}_{cls1}_uniq.txt')


if __name__ == '__main__':
  for limit in LIMITS:
    make_stats('char', limit, list)
    make_stats('word', limit, jieba.lcut_for_search)

    diff_vocab('char', limit, ['train', 'test'])
    diff_vocab('word', limit, ['train', 'test'])

    uniq_vocab('char', limit, ['train', 'test'])
    uniq_vocab('word', limit, ['train', 'test'])
