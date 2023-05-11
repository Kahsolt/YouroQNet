#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/03 

from pathlib import Path
BASE_PATH = Path(__file__).parent
import sys ; sys.path.append(str(BASE_PATH))
try: from run_quantum import *
except: pass

from subprocess import Popen


def train():
  ''' call for preprocess & training '''
  
  # preprocess
  PREPROCESS_SCRIPT = BASE_PATH / 'mk_vocab.py'
  assert PREPROCESS_SCRIPT.exists()
  cmd = f'python {PREPROCESS_SCRIPT}'
  p = Popen(cmd, shell=True, encoding='utf-8', stdout=sys.stdout)
  p.wait()

  # train
  pass


def question1(fp: str) -> np.ndarray:
  ''' call for inference '''

  # load model ckpt

  # predict
  df = pd.read_csv(fp)
  pred = df[df.columns[0]].to_numpy()
  return np.random.randint(0, 4, size=pred.shape)


if __name__ == '__main__':
  train()
