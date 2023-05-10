#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/10 

import pandas as pd

from utils import *

df_to_set = lambda x: { tuple(it) for it in x.to_numpy().tolist() }
set_to_df = lambda x: pd.DataFrame(x, columns=COULMNS, index=None)


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
