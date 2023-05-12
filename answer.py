#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/03 

from pathlib import Path
from subprocess import Popen

import numpy as np
import pandas as pd

BASE_PATH = Path(__file__).parent
import sys ; sys.path.append(str(BASE_PATH))
from run_quantum import get_args, go_train, go_infer


def get_args_best_config():
  ''' yes your just tune everything manually... '''

  args = get_args()
  # preprocess
  args.analyzer   = 'kgram+'
  args.min_freq   = 5
  # model
  args.model      = 'Youro'
  args.length     = 32
  args.embed_dim  = 32
  # train
  args.batch_size = 32
  args.epochs     = 10
  args.lr         = 0.1
  return args


def train():
  ''' call for preprocess & train '''
  
  # preprocess
  PREPROCESS_SCRIPT = BASE_PATH / 'mk_vocab.py'
  assert PREPROCESS_SCRIPT.exists()
  cmd = f'python {PREPROCESS_SCRIPT}'
  p = Popen(cmd, shell=True, encoding='utf-8', stdout=sys.stdout)
  p.wait()

  # train
  args = get_args_best_config()
  go_train(args)


def question1(fp: str) -> np.ndarray:
  ''' call for inference '''

  # query data
  df = pd.read_csv(fp)
  T = df[df.columns[-1]].to_numpy()

  # inference
  args = get_args_best_config()
  args.batch_size = 24
  pred = go_infer(args, T)

  if 'dummy random':
    pred = df[df.columns[0]].to_numpy()
    return np.random.randint(0, 4, size=pred.shape)
  else:
    return np.asarray(pred, dtype=np.int32)


if __name__ == '__main__':
  train()
