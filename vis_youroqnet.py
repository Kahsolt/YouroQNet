#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/12 

from pathlib import Path
from traceback import print_exc

BASE_PATH = Path(__file__).parent
import sys ; sys.path.append(str(BASE_PATH))
from run_quantum import get_parser, get_args, go_infer, mode, Inferer

from utils import get_acc_f1, load_dataset, mean, SPLITS

# interactive demo for the YouroQNet text classifier

LABEL_NAMES = {
  0: 'Joy',
  1: 'Angry',
  2: 'Hate',
  3: 'Sad',
}

def run_infer_dataset(args):
  for split in SPLITS:
    T, Y = load_dataset(split)
    preds = go_infer(args, T)
    acc, f1s = get_acc_f1(preds, Y)

    print(f'[{split}]')
    print(f'  acc:', acc)
    print(f'  f1:', f1s)
    print(f'  f1_avg:', mean(f1s))


def run_infer_interactive(args):
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


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--inplace', action='store_true', help='tokenize all dataset splits inplace')
  args = get_args(parser)

  if args.inplace:
    run_infer_dataset(args)
  else:
    run_infer_interactive(args)
