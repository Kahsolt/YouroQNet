#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/15 

from run_quantum import *

# NOTE: the toy YouroQNet with toy dataset for conceptual verification and ablation study

SUFFIX = '_toy'


def go_all(args):
  global vocab

  # preprocess: see run_quantum.gen_dataloader()
  preproc_pack = tokenizer, aligner, word2id, _, PAD_ID = get_preprocessor_pack(args, vocab)
  id2word = {v: k for k, v in word2id.items()}

  def preprocess(data):
    T_batch, Y_batch = [], []
    for lbl, txt in data:
      T_batch.append(sent_to_ids (txt, preproc_pack))
      Y_batch.append(id_to_onehot(lbl, args.n_class))
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
  go_train(args, user_vocab_data=(vocab, trainset, testset), name_suffix=SUFFIX)
  plt.show()

  # inspect
  words = [id2word[id] for id in sorted(id2word.keys())]
  go_inspect(args, name_suffix=SUFFIX, words=words)
  plt.show()


if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--bias', default='neu', choices=['neu', 'pos', 'neg'], help='bias neutral words towards')
  parser.add_argument('--tri', action='store_true', help='3-clf of pos/neg/neu')
  args = get_args(parser)

  print('>> Configs for the toy YouorQNet is finetuned, fixed and hard-coded, for further theoretical study :)')
  print('>> Be careful if you wanna modify this code indeed!! (e.g. make backups)')

  # model
  args.SEC_rots  = 'RY'
  args.SEC_entgl = 'CNOT'
  args.CMC_rots  = 'RY'
  # train
  args.epochs    = 75

  # should be fixed
  args.analyzer  = 'user'
  args.binary    = True
  args.n_len     = 3
  args.min_freq  = 1
  args.n_class   = 2
  args.n_vote    = 1

  if 'bias test':
    # NOTE: ablation verify, when learning is successful (check the loss curve), there are some evidental phenomenon
    #   'neu': the unbaised normal setting, which side to bias depends on dataset
    #   'pos': bais neutral pronouns to positive group, likely 恨/讨厌 will be highlighted out
    #   'neg': bais neutral pronouns to negative group, likely 爱/喜欢 will be highlighted out
    bias = args.bias

    if bias == 'neu':
      baised_data = []
    if bias == 'pos':
      baised_data = [
        (0, '我你'),
        (0, '你啊'),
        (0, '啊我啊啊'),
        (0, '西瓜苹果'),
        (0, '你西瓜'),
        (0, '我苹果啊'),
      ]
    if bias == 'neg':
      baised_data = [
        (1, '我你'),
        (1, '你啊'),
        (1, '啊我啊啊'),
        (1, '西瓜苹果'),
        (1, '你西瓜'),
        (1, '我苹果啊'),
      ]

  if '3-clf test':
    if args.tri:
      args.binary    = False
      args.onehot    = True
      args.n_class   = 3
      args.SEC_rots  = 'RY,RZ'
      args.SEC_entgl = 'CNOT'
      args.CMC_rots  = 'RY,RZ'
      
      train_data_cls2 = [
        (2, '啊你我'),
        (2, '西瓜苹果'),
        (2, '你苹果'),
        (2, '我啊'),
      ]
      test_data_cls2 = [
        (2, '苹果啊'),
        (2, '西瓜我西瓜'),
        (2, '啊啊啊'),
        (2, '你我'),
      ]
    else:
      train_data_cls2 = []
      test_data_cls2  = []

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
  ] + baised_data + train_data_cls2
  test_data = [
    (0, '西瓜喜欢'),
    (0, '我喜欢喜欢'),
    (0, '苹果爱'),
    (0, '你爱我'),
    (1, '讨厌你'),
    (1, '讨厌苹果'),
    (1, '恨你'),
    (1, '啊恨啊'),
  ] + test_data_cls2
  vocab = { w: sum(txt.count(w) for _, txt in train_data) for w in words }

  go_all(args)
