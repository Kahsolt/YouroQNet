#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/03 

from run_quantum import *


def train():
  """
  Please ensure that your code can run correctly and completely in IDE
  :return:
  """
  pass


def question1(fp: str) -> np.ndarray:
  """
  Please execute your model here to generate predictions for the validation set, which is not publicly
  available. The predictions should be returned in the form of an array. These returned values will be utilized to
  compute the F1-score and will serve as the criteria for evaluation.
  :param validation_set_name: string, dataset's file name.
  :return: ndarray, like np.array([0,1,2,3,1,1,2,0,3])
  """
  df = pd.read_csv(fp)
  pred = np.array([])
  return pred


if __name__ == '__main__':
  question1('data/test.csv')
