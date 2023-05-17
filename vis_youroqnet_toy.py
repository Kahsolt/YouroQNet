#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/15 

from run_quantum import *

# NOTE: the toy YouroQNet with toy dataset for conceptual verification and ablation study

SUFFIX = '_toy'

# NOTE: ablation verify, when learning is successful (check the loss curve), there are some evidental phenomenon
#   'neu': the unbaised normal setting, which side to bias depends on dataset
#   'pos': bais neutral pronouns to positive group, likely 恨/讨厌 will be highlighted out
#   'neg': bais neutral pronouns to negative group, likely 爱/喜欢 will be highlighted out
setting = 'neu'

if setting == 'neu':
  baised_data = []
if setting == 'pos':
  baised_data = [
    (0, '我你'),
    (0, '你啊'),
    (0, '啊我啊啊'),
    (0, '西瓜苹果'),
    (0, '你西瓜'),
    (0, '我苹果啊'),
  ]
if setting == 'neg':
  baised_data = [
    (1, '我你'),
    (1, '你啊'),
    (1, '啊我啊啊'),
    (1, '西瓜苹果'),
    (1, '你西瓜'),
    (1, '我苹果啊'),
  ]

words = {
  # positive (leading to class 0)
  '爱', '喜欢',
  # negative (leading to class 1)
  '恨', '讨厌',
  # neutral
  '啊', '我', '你', '苹果', '西瓜',
}
train_data = [
  (0, '我爱你'),
  (0, '我喜欢苹果'),
  (0, '苹果啊喜欢'),
  (0, '你爱西瓜'),
  (1, '你讨厌我'),
  (1, '讨厌西瓜苹果'),
  (1, '你讨厌苹果'),
  (1, '我恨啊恨'),
] + baised_data
test_data = [
  (0, '西瓜喜欢'),
  (0, '我喜欢喜欢'),
  (0, '苹果爱'),
  (0, '你爱我'),
  (1, '讨厌你'),
  (1, '讨厌苹果'),
  (1, '恨你'),
  (1, '啊恨啊'),
]
vocab = { w: sum(txt.count(w) for _, txt in train_data) for w in words }


def go_all(args):
  global vocab

  # preprocess: see run_quantum.gen_dataloader()
  tokenizer, aligner, word2id, PAD_ID = get_preprocessor_pack(args, vocab)
  id2word = {v: k for k, v in word2id.items()}

  def preprocess(data):
    T_batch, Y_batch = [], []
    for lbl, txt in data:
      T_batch.append(np.asarray([word2id.get(w, PAD_ID) for w in aligner(tokenizer(txt))]))
      Y_batch.append(np.eye(args.n_class)[lbl])
    return [np.stack(e, axis=0).astype(np.int32) for e in [T_batch, Y_batch]]

  # dataset
  trainset = preprocess(train_data)
  validset = preprocess(test_data)

  if not 'preview_dataset':
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

  # train
  trainset = [e[1] for e in train_data], [e[0] for e in train_data]
  testset  = [e[1] for e in test_data],  [e[0] for e in test_data]
  go_train(args, (vocab, trainset, testset), name_suffix=SUFFIX)
  plt.show()

  # inspect
  words = [id2word[id] for id in sorted(id2word.keys())]
  go_inspect(args, name_suffix=SUFFIX, words=words)
  plt.show()


if __name__ == '__main__':
  args = get_args()

  print('>> Configs for the toy YouorQNet is finetuned, fixed and hard-coded, for further theoretical study :)')
  print('>> Be careful if you wanna modify this code indeed!! (e.g. make backups)')

  # model
  args.n_repeat   = 1
  args.embed_var  = 0.2
  args.embed_norm = 1
  args.SEC_rots   = 'RY'
  args.SEC_entgl  = 'CNOT'
  args.CMC_rots   = 'RY'
  # train
  args.epochs     = 75
  args.batch_size = 1
  args.lr         = 0.01
  args.optim      = 'Adam'
  args.grad_meth  = 'fd'
  args.grad_dx    = 0.01

  # should be fixed
  args.analyzer      = 'user'
  args.model         = 'Youro'
  args.n_len         = 3
  args.min_freq      = 1
  args.n_class       = 2
  args.n_vote        = 1
  args.slog_interval = 10
  args.log_interval  = 50

  go_all(args)
