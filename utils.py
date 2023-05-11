#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/05 

import sys
from pathlib import Path
from time import time
import logging
from logging import Logger
from traceback import print_exc
from typing import List, Tuple

import pandas as pd
import numpy as np

RANDSEED = 114514

DATA_PATH = Path('data') if sys.platform == 'win32' else Path('.')
LOG_PATH  = Path('log') ; LOG_PATH.mkdir(exist_ok=True)

''' utils '''

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

def timer(fn):
  def wrapper(*args, **kwargs):
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.2f}s')
    return r
  return wrapper


''' dataset & text'''

if 'consts for dataset':
  N_CLASS = 4
  COULMNS = ['label', 'text']
  SPLITS  = ['train', 'test', 'valid'] if sys.platform == 'win32' else ['train', 'test']

def load_dataset(split:str, normalize:bool=False, fp:Path=None) -> Tuple[np.ndarray, List[str]]:
  ''' `fp` overrides the default filepath '''

  fp_norm = fp or DATA_PATH / f'{split}_cleaned.csv'
  if normalize and fp_norm.exists():
    print(f'load cleaned {split} from cache {fp_norm}')
    return load_dataset(split, False, fp_norm)

  fp = fp or DATA_PATH / f'{split}.csv'
  df = pd.read_csv(fp)
  c_lbl, c_txt = df.columns[0], df.columns[-1]
  if split == 'valid':    # the whole valid set is too large
    df = pd.concat([df_cls.sample(n=1000, random_state=RANDSEED) for _, df_cls in df.groupby(c_lbl)])
  Y = df[c_lbl].to_numpy().astype(np.int32)
  T = df[c_txt].to_numpy().tolist()
  if normalize: T = clean_text(T)
  return T, Y


if 'consts for text':
  from re import compile as Regex
  from string import printable

  # https://fuhaoku.net/blocks
  # https://www.qqxiuzi.cn/zh/hanzi-unicode-bianma.php
  # https://zh.wikipedia.org/wiki/Help:%E9%AB%98%E7%BA%A7%E5%AD%97%E8%AF%8D%E8%BD%AC%E6%8D%A2%E8%AF%AD%E6%B3%95

  ASCII = { chr(i) for i in range(128) }
  HF_CP = (0xFF00, 0xFFEF)  # 全角/半角符号
  JP_CP = (0x3040, 0x30FF)  # 日语假名
  ZH_CP = (0x4E00, 0x9FA5)  # 汉字

  IGNORE_CHARS = {
    '\ufeff',
    '"', '“', '”',
  } | (ASCII - set(printable))
  R_DEADCHAR   = Regex('|'.join(IGNORE_CHARS))
  R_WHITESPACE = Regex(r'\s+')
  R_NUMBER     = Regex(r'[-+]?[0-9]+')
  R_CJK_PERIOD = Regex('\u3002|\uFF61')   # '。'
  R_CJK_PUASE  = Regex('\u3001')          # '、'

  # TODO: hard-coded
  STOP_WORDS_CHAR = ['，', ' ', '的', '。', '.', '！', '我', '了', '是', '不', '一']
  STOP_WORDS_WORD = ['，', ' ', '的', '。', '！', '了', '我']

  def wchar_to_char(line:str) -> str:
    # ord('！') - ord('!') == 65248
    to_latin = lambda c: chr(ord(c) - 65248) if HF_CP[0] <= ord(c) <= HF_CP[1] else c
    return ''.join([to_latin(c) for c in list(line)])

  def try_concat(line:str) -> str:
    is_alphanum = lambda ch: 'a' <= ch <= 'z' or '0' <= ch <= '9'
    segs = [seg for seg in line.split(' ') if seg]
    r = segs[0]
    for seg in segs[1:]:
      if is_alphanum(r[-1]) and is_alphanum(seg[0]):
        r += ' '
      r += seg
    return r

  def fold_triple(line:str) -> str:
    chars = list(line)
    r = chars[:2]
    for i in range(2, len(chars)):
      if chars[i] == r[-1] == r[-2]: continue
      r.append(chars[i])
    return ''.join(r)

def clean_text(texts:List[str]) -> List[str]:
  def _process(s:str) -> str:
    try:
      s = R_DEADCHAR.sub('', s)
      s = R_CJK_PERIOD.sub('.', s)
      s = R_CJK_PUASE.sub(',', s)
      s = wchar_to_char(s)
      s = s.lower()
      s = R_WHITESPACE.sub(' ', s)
      s = R_NUMBER.sub('0', s)
      s = try_concat(s)
      s = fold_triple(s)
    except: print_exc()
    return s
  return [_process(line) for line in texts]


''' metric '''

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
