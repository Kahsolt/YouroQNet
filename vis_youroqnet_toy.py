#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/15 

from run_quantum import *

# NOTE: the mini YouroQNet for conceptual verification

vocab = {
  '我': 2,
  '你': 3,
  '爱': 2,
  '喜欢': 1,
  '讨厌': 3,
  '苹果': 2,
  '西瓜': 2,
}
train_data = [
  (0, '我爱你'),
  (0, '我喜欢苹果'),
  (0, '你爱西瓜'),
  (1, '你讨厌我'),
  (1, '讨厌西瓜'),
  (1, '你讨厌苹果'),
]
valid_data = [
  (0, '喜欢'),
  (0, '喜欢喜欢我'),
  (0, '苹果爱'),
  (0, '你爱我'),
  (1, '讨厌你'),
  (1, '讨厌讨厌'),
]


def preview_dataset(args):
  tokenizer = make_tokenizer(vocab)
  word2id = get_word2id(args, list(vocab.keys()))
  PAD_ID = word2id.get(args.pad, -1)
  aligner = lambda x: align_words(x, args.length, args.pad)

  def preprocess(data):
    X, Y = [], []
    for lbl, txt in data:
      Y.append(np.eye(args.n_class)[lbl])
      X.append(np.asarray([word2id.get(w, PAD_ID) for w in aligner(tokenizer(txt))]))
    return [np.stack(e, axis=0).astype(np.int32) for e in [X, Y]]

  trainset = preprocess(train_data)
  validset = preprocess(valid_data)

  print('word2id:')
  print(word2id)
  print('trainset:')
  print(trainset[0])
  print(trainset[1])
  print('validset:')
  print(validset[0])
  print(validset[1])


def go_train(args):
  # logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  h = logging.StreamHandler()
  h.setFormatter(fmt=logging.Formatter('%(message)s'))
  logger.addHandler(h)

  # symbols (codebook)
  args.n_vocab = len(vocab) + 1  # <PAD>

  # data
  trainset = [e[1] for e in train_data], [e[0] for e in train_data]
  validset = [e[1] for e in valid_data], [e[0] for e in valid_data]
  train_loader = gen_dataloader(args, trainset, vocab, shuffle=True)
  valid_loader = gen_dataloader(args, validset, vocab)

  # model & optimizer & loss
  model, creterion = get_model_and_creterion(args)
  args.param_cnt = sum([p.size for p in model.parameters() if p.requires_grad])

  print(f'hparams: {pformat(vars(args))}')

  if args.optim == 'SGD':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
  elif args.optim == 'Adam':
    optimizer = Adam(model.parameters(), lr=args.lr)

  # train
  losses_and_accs = train(args, model, optimizer, creterion, train_loader, valid_loader, logger)
  
  # plot
  plot_loss_and_acc(losses_and_accs, TMP_PATH / 'vis_youroqnet_toy.png', title='YouroQNet toy')


if __name__ == '__main__':
  args = get_args()
  # tunable
  args.repeat = 4
  args.epochs = 150
  args.batch_size = 1
  args.lr = 0.01
  args.grad_meth = 'fd'
  args.grad_dx = 0.01

  # fixed
  args.model = 'Youro'
  args.length = 3
  args.min_freq = 1
  args.n_class = 2
  args.n_vote = 1
  args.log_interval = 10
  args.log_reset_interval = 50
  args.test_interval = 50
  
  preview_dataset(args)
  go_train(args)
