#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/15 

from run_quantum import *
from run_baseline_vq import get_preprocessor_pack

# NOTE: the toy dataset on the minimal YouroQNet for conceptual verification

SUFFIX = '_toy'

words = {
  # positive (leading to class 0)
  '爱', '喜欢',
  # negetive (leading to class 1)
  '恨', '讨厌',
  # neutral
  '我', '你', '苹果', '西瓜',
}
train_data = [
  (0, '我爱你'),
  (0, '我喜欢苹果'),
  (0, '你爱西瓜'),
  (1, '你讨厌我'),
  (1, '讨厌西瓜'),
  (1, '你讨厌苹果'),
]
test_data = [
  (0, '喜欢'),
  (0, '喜欢喜欢我'),
  (0, '苹果爱'),
  (0, '你爱我'),
  (1, '讨厌你'),
  (1, '讨厌讨厌'),
]
vocab = { w: sum(txt.count(w) for _, txt in train_data) for w in words }


def preview_dataset(args):
  ''' see run_quantum.gen_dataloader() '''
  tokenizer, aligner, word2id, PAD_ID = get_preprocessor_pack(args, vocab)
  id2word = {v: k for k, v in word2id.items()}

  def preprocess(data):
    T_batch, Y_batch = [], []
    for lbl, txt in data:
      T_batch.append(np.asarray([word2id.get(w, PAD_ID) for w in aligner(tokenizer(txt))]))
      Y_batch.append(np.eye(args.n_class)[lbl])
    return [np.stack(e, axis=0).astype(np.int32) for e in [T_batch, Y_batch]]

  trainset = preprocess(train_data)
  validset = preprocess(test_data)

  print('=' * 72)
  print('vocab:', vocab)
  print('word2id:', word2id)
  print('id2word:', id2word)
  print('trainset:')
  print(trainset[0])
  print(trainset[1])
  print('validset:')
  print(validset[0])
  print(validset[1])
  print('=' * 72)


def go_train_proxy(args):
  global vocab

  trainset = [e[1] for e in train_data], [e[0] for e in train_data]
  testset  = [e[1] for e in test_data],  [e[0] for e in test_data]
  go_train(args, (vocab, trainset, testset), name_suffix=SUFFIX)


def go_inspect_proxy(args):
  go_inspect(args, name_suffix=SUFFIX)


if __name__ == '__main__':
  args = get_args()
  # tunable
  args.epochs     = 100
  args.batch_size = 1
  args.lr         = 0.01
  args.grad_meth  = 'fd'
  args.grad_dx    = 0.01

  # fixed
  args.analyzer      = 'user'
  args.model         = 'Youro'
  args.length        = 3
  args.min_freq      = 1
  args.n_class       = 2
  args.n_vote        = 1
  args.slog_interval = 10
  args.log_interval  = 50
  args.test_interval = 50
  
  if args.inspect:
    go_inspect_proxy(args)
    exit(0)

  preview_dataset(args)
  go_train_proxy(args)
