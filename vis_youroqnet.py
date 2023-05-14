#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/12 

from pathlib import Path
from traceback import print_exc

BASE_PATH = Path(__file__).parent
import sys ; sys.path.append(str(BASE_PATH))
from run_quantum import get_args, go_infer, mode, Inferer

LABEL_NAMES = {
  0: 'Joy',
  1: 'Angry',
  2: 'Hate',
  3: 'Sad',
}

args = get_args()
inferer: Inferer = go_infer(args)

try:
  while True:
    T = input('>> input a sentence: ').strip()
    if not T: continue
    
    votes = inferer(T)
    final = mode(votes)
    print(f'<< pred: {votes} => {final} => {LABEL_NAMES[final]}')
except KeyboardInterrupt:
  pass
except:
  print_exc()
