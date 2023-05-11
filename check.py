#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/11 

# modified from the official judge script

import sys
import io
import numpy as np
import pandas as pd

from answer import question1

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def f1_score(result, target, label) -> float:
  solution_label = (result == label).astype(int)
  target_label   = (target == label).astype(int)
  tp = np.sum(solution_label * target_label)
  fp = np.sum(solution_label * (1 - target_label))
  fn = np.sum((1 - solution_label) * target_label)

  p = tp / (tp + fp + 1e-7)
  f = tp / (tp + fn + 1e-7)

  return np.round(2 * (p * f) / (p + f), 3)


def score(target, result) -> float:
  s = 0
  for label in [0, 1, 2, 3]:
    s += f1_score(result, target, label)
    print('each score', f1_score(result, target, label))
  return s * 15


def get_target_label(fp:str) -> np.ndarray:
  df = pd.read_csv(fp)
  return df[df.columns[0]].to_numpy()


if __name__ == "__main__":
  for split in ['train', 'test', 'valid']:
    print(f'[{split}]')
    fp = f'data/{split}.csv'
    target = get_target_label(fp)
    solution = question1(fp)
    print('final score:', score(target, solution))
    print()
