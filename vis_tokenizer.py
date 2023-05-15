#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/10 

from mk_vocab import *
from utils import DATA_PATH, SPLITS, clean_text


def run_tokenize_dataset():
  for split in SPLITS:
    T, _ = load_dataset(split)    # dataset is normalize by default
    tokenizer = make_tokenizer()  # default to kgram

    fp = DATA_PATH / f'{split}_tokenized.txt'
    with open(fp, 'w', encoding='utf-8') as fh:
      for t in T:
        segs = tokenizer(t, top_k=None)
        fh.write(' '.join(segs) + '\n')
    print(f'>> save to {fp}...')


def run_tokenize_interactive():
  tokenizer = make_tokenizer()  # default to kgram
  try:
    while True:
      t = input('input a sentence: ')
      t = clean_text(t)         # call normalize manually
      for logp, segs in tokenizer(t, top_k=-1):
        print(f'[{logp:.3f}]', ' '.join(segs))
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--inplace', action='store_true', help='tokenize all dataset splits inplace')
  args = parser.parse_args()

  if args.inplace:
    run_tokenize_dataset()
  else:
    run_tokenize_interactive()
