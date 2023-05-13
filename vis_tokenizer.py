#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/10 

from mk_vocab import *
from utils import SPLITS, clean_texts


def run_tokenize(split:str):
  T, _ = load_dataset(split)    # dataset is normalize by default
  tokenizer = make_tokenizer()  # default to kgram
  for t in T:
    for logp, segs in tokenizer(t, top_k=-1):
      print(f'[{logp:.3f}]', ' '.join(segs))
    input()


def run_tokenize_ion():
  tokenizer = make_tokenizer()  # default to kgram
  try:
    while True:
      t = input('input a sentence: ')
      t = clean_texts([t])[0]   # call normalize manually
      for logp, segs in tokenizer(t, top_k=-1):
        print(f'[{logp:.3f}]', ' '.join(segs))
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-D', '--split', default=None, choices=SPLITS+[None], help='dataset split')
  args = parser.parse_args()

  if args.split:
    run_tokenize(args.split)
  else:
    run_tokenize_ion()
