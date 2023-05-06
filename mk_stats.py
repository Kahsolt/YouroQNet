#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/03 

from pathlib import Path
from functools import reduce
from copy import deepcopy
from collections import Counter, defaultdict, OrderedDict
from typing import List, Dict, Callable

import jieba
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

DATA_PATH = Path('data')
LOG_PATH  = Path('log') ; LOG_PATH.mkdir(exist_ok=True)

N_CLASS = 4
COULMNS = ['label', 'text']
SPLITS  = ['train', 'test', 'valid']

df_to_set = lambda x: { tuple(it) for it in x.to_numpy().tolist() }
set_to_df = lambda x: pd.DataFrame(x, columns=COULMNS, index=None)


def load_vocab(fp:str) -> Dict[str, int]:
  with open(fp, encoding='utf-8') as fh:
    lines = fh.read().rstrip().split('\n')
  val_cnt = [line.split('\t') for line in lines]
  return {v: int(c) for v, c in val_cnt}


def dump_vocab(voc:Dict[str, int], fp:str):
  wc = WordCloud(font_path='simhei.ttf', height=1600, width=2048, background_color='white')
  wc.fit_words(voc)
  wc.to_file(Path(fp).with_suffix('.png'))

  with open(fp, 'w', encoding='utf-8') as fh:
    for v, c in voc.items():
      fh.write(f'{v}\t{c}\n')


def write_stats(items:List[str], name:str, subfolder:str=''):
  out_dp = LOG_PATH / subfolder / 'stats'
  out_dp.mkdir(exist_ok=True, parents=True)

  pairs = sorted([(c, v) for v, c in Counter(items).items()], reverse=True)
  dump_vocab(OrderedDict([(v, c) for c, v in pairs]), out_dp / f'vocab_{name}.txt')

  with open(out_dp / f'stats_{name}.txt', 'w', encoding='utf-8') as fh:
    fh.write(f'words: {len(items)}\n')
    fh.write(f'vocab: {len(pairs)}\n')

  plt.clf()
  plt.plot([c for c, v in pairs])
  plt.suptitle(f'freq_{name}')
  plt.savefig(out_dp / f'freq_{name}.png')


def write_lens(lens:List[int], name:str, subfolder:str=''):
  out_dp = LOG_PATH / subfolder / 'stats'
  out_dp.mkdir(exist_ok=True, parents=True)

  pairs = sorted([(v, c) for v, c in Counter(lens).items()], reverse=True)
  x = [v for v, c in pairs]
  y = [c for v, c in pairs]

  plt.clf()
  plt.plot(x, y)
  plt.suptitle(f'len_{name}')
  plt.savefig(out_dp / f'len_{name}.png')


def make_stats(kind:str, line_parser:Callable):
  words_all = []
  for split in SPLITS:
    df = pd.read_csv(DATA_PATH / f'{split}.csv')
    label, text = df['label'].to_numpy().astype(int), df['text'].to_numpy()

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
    words_split = reduce(lambda x, y: x.extend(y) or x, words_cls.values(), [])
    write_stats(words_split, split, subfolder=kind)
    lens_split = reduce(lambda x, y: x.extend(y) or x, lens_cls.values(), [])
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


def make_validset():
  ''' We guess that valid set is right the complementary of given train/test set ;) '''

  df_all = pd.read_csv(DATA_PATH / 'simplifyweibo_4_moods.csv')
  print(f'all:\t{len(df_all)}')

  df_train = pd.read_csv(DATA_PATH / 'train.csv') 
  df_test  = pd.read_csv(DATA_PATH / 'test.csv')
  print(f'train:\t{len(df_train)}')
  print(f'test:\t{len(df_test)}')
  
  if 'assert_train_test_not_overlap':
    rec1, rec2 = df_to_set(df_train), df_to_set(df_test)
    assert rec1.isdisjoint(rec2) and len(rec1.union(rec2)) == len(rec1) + len(rec2)

  df_known = pd.concat([df_train, df_test])

  df_valid = set_to_df(df_to_set(df_all) - df_to_set(df_known))
  df_valid.to_csv(DATA_PATH / 'valid.csv', index=None)
  print(f'valid:\t{len(df_valid)}')

  if 'see train + test + valid - all':
    df_ttv = pd.concat([df_known, df_valid])
    df_unknown = set_to_df(df_to_set(df_ttv) - df_to_set(df_all))
    df_unknown.to_csv(DATA_PATH / 'unknown.csv', index=None)
    print(f'unknown:\t{len(df_unknown)}')


if __name__ == '__main__':
  make_validset()

  make_stats('char', list)
  make_stats('word', jieba.lcut_for_search)

  diff_vocab('char')
  diff_vocab('word')

  uniq_vocab('char')
  uniq_vocab('word')
