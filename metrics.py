#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/03 

import numpy as np


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
