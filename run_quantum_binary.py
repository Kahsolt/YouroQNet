#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/09 

from run_quantum import *

# binary clf for each class


def gen_dataloader(args, dataset:Dataset, vocab:Vocab, shuffle:bool=False) -> Dataloader:
  preproc_pack = get_preprocessor_pack(args, vocab)

  def iter_by_batch() -> Tuple[NDArray, NDArray]:
    nonlocal args, dataset, preproc_pack, shuffle

    T, Y = dataset
    N = len(Y)
    indexes = list(range(N))
    if shuffle: random.shuffle(indexes) 

    for i in range(0, N, args.batch_size):
      T_batch, Y_batch = [], []
      for j in range(args.batch_size):
        if i + j >= N: break
        idx = indexes[i + j]
        lbl = np.int32(Y[idx] == args.tgt_cls)
        T_batch.append(sent_to_ids (T[idx], preproc_pack))
        Y_batch.append(id_to_onehot(lbl, args.n_class))

      if len(T_batch) == args.batch_size:
        yield [np.stack(e, axis=0).astype(np.int32) for e in [T_batch, Y_batch]]
  
  return iter_by_batch    # return a DataLoader generator


def go_train(args, user_vocab_data:Tuple[Vocab, Dataset, Dataset]=None, name_suffix:str=''):
  # configs
  args.expname = f'{args.analyzer}_{args.model}{name_suffix}'
  out_dp: Path = LOG_PATH / args.analyzer / f'{args.model}{name_suffix}'
  out_dp.mkdir(exist_ok=True, parents=True)
  args.out_dp = str(out_dp)
  logger = get_logger(out_dp / 'run.log', mode='w')

  # symbols (codebook)
  if user_vocab_data:
    vocab, trainset, testset = user_vocab_data
  else:
    vocab = get_vocab(args)
  args.n_vocab = len(vocab) + 1  # <PAD>

  # data
  if user_vocab_data:
    train_loader = gen_dataloader(args, trainset, vocab, shuffle=True)
    test_loader  = gen_dataloader(args, testset,  vocab)
  else:
    train_loader = gen_dataloader(args, load_dataset('train'), vocab, shuffle=True)
    test_loader  = gen_dataloader(args, load_dataset('test'),  vocab)
    if MODE_DEV:
      valid_loader = gen_dataloader(args, load_dataset('valid'), vocab)

  # model & optimizer & loss
  model, criterion = get_model_and_criterion(args)    # criterion accepts onehot label as truth
  args.param_cnt = sum([p.size for p in model.parameters() if p.requires_grad])
  
  if args.optim == 'SGD':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
  elif args.optim == 'Adam':
    optimizer = Adam(model.parameters(), lr=args.lr)

  # info
  logger.info(f'hparam: {pformat(vars(args))}')

  # train
  losses_and_accs = train(args, model, optimizer, criterion, train_loader, test_loader, logger)
  params = model.m_para.to_numpy()

  # plot
  plot_loss_and_acc(losses_and_accs, out_dp / PLOT_FILE, title=args.expname)
  
  # save & load
  save_ckpt(model, out_dp / MODEL_FILE)
  load_ckpt(model, out_dp / MODEL_FILE)
  
  # eval
  result = { 
    'hparam': vars(args), 
    'ansatz': get_ansatz(args, params).tolist(),
    'embed': get_embed(args, params).tolist(),
    'scores': {},
  }
  for split in SPLITS:
    datat_loader = locals().get(f'{split}_loader')
    if datat_loader is None: continue     # ignore if valid_loader not exists
    logger.info(f'testing split {split} ...')
    loss, acc, f1 = test(args, model, criterion, datat_loader, logger)
    result['scores'][split] = {
      'loss': loss,
      'acc':  acc,
      'f1':   f1,
    }
  json_dump(result, out_dp / TASK_FILE)


if __name__ == '__main__':
  args = get_args()

  args.n_class = 2
  args.tgt_cls = 0      # [0, 1, 2, 3]

  go_train(args, name_suffix='binary')
  go_inspect(args)
