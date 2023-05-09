#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/05 

from pathlib import Path
import logging
from logging import Logger
from typing import List, Tuple

import pandas as pd
import numpy as np

RANDSEED = 114514

DATA_PATH = Path('data')
LOG_PATH  = Path('log') ; LOG_PATH.mkdir(exist_ok=True)

N_CLASS = 4
COULMNS = ['label', 'text']
SPLITS  = ['train', 'test', 'valid']

# TODO: hard-coded
STOP_WORDS_CHAR = ['，', ' ', '的', '。', '.', '！', '我', '了', '是', '不', '一']
STOP_WORDS_WORD = ['，', ' ', '的', '。', '！', '了', '我']


def get_logger(fp:str, mode:str='a') -> Logger:
  FORMATTER = logging.Formatter('%(message)s')
  logger = logging.getLogger('run')
  logger.setLevel(logging.DEBUG)
  h = logging.FileHandler(fp, mode, encoding='utf-8')
  h.setLevel(logging.DEBUG)
  h.setFormatter(FORMATTER)
  logger.addHandler(h)
  h = logging.StreamHandler()
  h.setLevel(logging.DEBUG)
  h.setFormatter(FORMATTER)
  logger.addHandler(h)
  return logger


def load_dataset(split:str) -> Tuple[np.ndarray, List[str]]:
  df = pd.read_csv(DATA_PATH / f'{split}.csv')
  c_lbl, c_txt = df.columns[0], df.columns[-1]
  if split == 'valid': df = pd.concat([df_cls.sample(n=1000, random_state=RANDSEED) for _, df_cls in df.groupby(c_lbl)])
  Y = df[c_lbl].to_numpy().astype(np.int32)
  T = df[c_txt].to_numpy().tolist()
  return T, Y


def clean_text(texts:List[str]) -> List[str]:
  r = []
  for line in texts:
    r.append(' '.join([e.strip() for e in list(line) if e.strip()]))
  return r


def confusion_matrix(pred, truth, num_class:int=None):
  num_class = num_class or max(truth) + 1

  cmat = np.zeros([num_class, num_class], dtype=np.int32)
  for p, t in zip(pred, truth): cmat[t, p] += 1
  return cmat


if __name__ == '__main__':
  N = 1024
  num_class = 4
  pred  = np.random.randint(0, num_class, size=[N])
  truth = np.random.randint(0, num_class, size=[N])
  cmat = confusion_matrix(pred, truth)
  print(cmat)
