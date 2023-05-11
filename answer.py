#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/03 

from .run_quantum import *


def train():
  pass


def question1(fp: str) -> np.ndarray:
  df = pd.read_csv(fp)
  pred = df[df.columns[0]].to_numpy()
  return np.random.randint(0, 4, size=pred.shape)


if __name__ == '__main__':
  fp = 'data/valid.csv'
  print(question1(fp))
