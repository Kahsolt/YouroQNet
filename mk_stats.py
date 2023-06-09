#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/03 

from functools import reduce
from copy import deepcopy
from collections import Counter, defaultdict
from typing import List, Callable

import jieba
import matplotlib.pyplot as plt

from utils import LOG_PATH, SPLITS, N_CLASS, load_dataset, savefig
from mk_vocab import load_vocab, dump_vocab, sort_vocab


def write_stats(tokens:List[str], name:str, subfolder:str=''):
  out_dp = LOG_PATH / subfolder / 'stats'
  out_dp.mkdir(exist_ok=True, parents=True)

  voc = sort_vocab(Counter(tokens))
  dump_vocab(voc, out_dp / f'vocab_{name}.txt')

  with open(out_dp / f'stats_{name}.txt', 'w', encoding='utf-8') as fh:
    fh.write(f'words: {len(tokens)}\n')
    fh.write(f'vocab: {len(voc)}\n')

  plt.clf()
  plt.plot(voc.values())
  plt.suptitle(f'freq_{name}')
  savefig(out_dp / f'freq_{name}.png')


def write_lens(lens:List[int], name:str, subfolder:str=''):
  out_dp = LOG_PATH / subfolder / 'stats'
  out_dp.mkdir(exist_ok=True, parents=True)

  pairs = sorted([(v, c) for v, c in Counter(lens).items()], reverse=True)
  x = [v for v, c in pairs]
  y = [c for v, c in pairs]

  plt.clf()
  plt.plot(x, y)
  plt.suptitle(f'len_{name}')
  savefig(out_dp / f'len_{name}.png')


def make_stats(kind:str, line_parser:Callable):
  words_all = []
  for split in SPLITS:
    text, label = load_dataset(split)

    words_cls = defaultdict(list)
    lens_cls  = defaultdict(list)
    for cls, line in zip(label, text):
      tokens = line_parser(line) ; assert '' not in tokens
      words_cls[cls].extend(tokens)
      lens_cls [cls].append(len(tokens))

    # class-wise
    for cls in words_cls.keys():
      write_stats(words_cls[cls], f'{split}_{cls}', subfolder=kind)
      write_lens (lens_cls [cls], f'{split}_{cls}', subfolder=kind)
    # split-wise
    words_split = reduce(lambda ret, words: ret.extend(words) or ret, words_cls.values(), [])
    write_stats(words_split, split, subfolder=kind)
    lens_split = reduce(lambda ret, lens: ret.extend(lens) or ret, lens_cls.values(), [])
    write_lens(lens_split, split, subfolder=kind)

    words_all.extend(words_split)

  # dataset-wise
  write_stats(words_all, 'all', subfolder=kind)


def diff_vocab(kind:str):
  out_dp = LOG_PATH / kind / 'stats'

  vocabs = { split: load_vocab(out_dp / f'vocab_{split}.txt') for split in SPLITS }

  for split1 in SPLITS:
    for split2 in SPLITS:
      if split1 == split2: continue

      voc1 = deepcopy(vocabs[split1])
      voc2 = vocabs[split2]
      for key in voc2:
        if key in voc1:
          del voc1[key]

      dump_vocab(voc1, out_dp / f'vocab_{split1}-{split2}.txt')


def uniq_vocab(kind:str):
  out_dp = LOG_PATH / kind / 'stats'

  for split in SPLITS:
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
  LINE_PARSER = {
    'char': list,
    'word': jieba.lcut_for_search,
  }
  for level in ['char', 'word']:
    print(f'>> making stats for {level} level ...')
    make_stats(level, LINE_PARSER[level])

  for level in ['char', 'word']:
    print(f'>> diff vocab for {level} level ...')
    diff_vocab(level)

  for level in ['char', 'word']:
    print(f'>> uniq vocab for {level} level...')
    uniq_vocab(level)
