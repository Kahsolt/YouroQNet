#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/05 

from run_baseline_vq import *

import torch
from torch.nn import Module
from torch.nn import Conv1d, Conv2d, MaxPool1d, MaxPool2d, Embedding, Linear, Dropout, ReLU
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# NOTE: VQNet impl of TextCNN does not work, we use torch instead

to_tensor = lambda *xs: tuple(torch.from_numpy(x).long() for x in xs) if len(xs) > 1 else torch.from_numpy(xs[0])
argmax    = lambda x: x.argmax(dim=-1)

def load_ckpt(model:Module, fp:str):
  model.load_state_dict(torch.load(fp))

def save_ckpt(model:Module, fp:str):
  torch.save(model.state_dict(), fp)


class TextModel(Module):

  def __init__(self, args):
    super().__init__()

    self.args = args
    self.embed = Embedding(args.n_vocab, args.embed_dim)  # [K, D]

    if hasattr(args, 'agg'):
      if args.agg == 'avg': self.agg = torch.mean
      if args.agg == 'max': self.agg = lambda *args: torch.max(*args)[0]


class TextCNN(TextModel):

  def __init__(self, args):
    super().__init__(args)
  
    if args.type == '1d':
      self.conv1 = Conv1d(args.embed_dim, args.dim//2, 3, 1, 'same')
      self.pool  = MaxPool1d(2, 2)
      self.conv2 = Conv1d(args.dim//2, args.dim, 3, 1, 'same')
    if args.type == '2d':
      self.conv1 = Conv2d(1, args.dim//2, (3, 3), (1, 1), 'same')
      self.pool  = MaxPool2d((2, 2), (2, 2))
      self.conv2 = Conv2d(args.dim//2, args.dim, (3, 3), (1, 1), 'same')

    self.act = ReLU()
    self.drp = Dropout(0.35)
    self.fc  = Linear(args.dim, args.n_class)

  def forward(self, x:QTensor):
    o = self.embed(x)             # [B, L=32, D=32]
    o = torch.permute(o, [0, 2, 1])  # [B, C=32, L=32]

    if args.type == '1d':
      o = self.conv1(o)           # [B, C=40, L=32]
      o = self.pool (o)           # [B, C=40, L=16]
      o = self.act  (o)
      o = self.conv2(o)
      o = self.pool (o)           # [B, C=80, L=8]
      o = self.act  (o)
      o = self.agg  (o, -1)       # [B, D=80]
      o = self.drp  (o)
      o = self.fc   (o)           # [B, NC=4]

    if args.type == '2d':
      o = torch.unsqueeze(o, 1)  # [B,  C=1, H=32, W=32]
      o = self.conv1(o)
      o = self.pool (o)           # [B, C=40, H=16, W=16]
      o = self.act  (o)
      o = self.conv2(o)
      o = self.pool (o)           # [B, C=80, H=8, W=8]
      o = self.act  (o)
      B, C, H, W = o.shape
      o = torch.reshape(o, (B, C, H * W))
      o = self.agg  (o, -1)       # [B, D=80]
      o = self.drp  (o)
      o = self.fc   (o)           # [B, NC=4]

    return o


def get_model(args) -> Module:
  name: str = args.model

  if name.startswith('cnn'):
    try:
      _, type, dim, agg = name.split('-')

      dim = int(dim) ; assert dim > 0
      assert type in ['1d', '2d']
      assert agg in ['avg', 'max']
    except:
      print_exc()
      raise ValueError('name pattern should follow: cnn-[type:str]-[dim:int]-[agg:str]')

    args.type = type
    args.dim = dim
    args.agg = agg
    return TextCNN(args)
  
  raise ValueError(name)


def test(args, model:Module, data_loader:Dataloader, logger:Logger) -> Score:
  Y_true, Y_pred = [], []

  model.eval()
  for X_np, Y_np in data_loader():
    X, Y = to_tensor(X_np, Y_np)
  
    logits = model(X)
    pred = argmax(logits)

    Y_true.extend(   Y.numpy().tolist())
    Y_pred.extend(pred.numpy().tolist())
  
  Y_true = np.asarray(Y_true, dtype=np.int32)
  Y_pred = np.asarray(Y_pred, dtype=np.int32)

  prec, recall, f1, _ = precision_recall_fscore_support(Y_true, Y_pred, average=None)
  cmat = confusion_matrix(Y_true, Y_pred)
  logger.info(f'>> prec: {prec}')
  logger.info(f'>> recall: {recall}')
  logger.info(f'>> f1: {f1}')
  logger.info('>> confusion_matrix:')
  logger.info(cmat)
  return prec, recall, f1, cmat

def valid(args, model:Module, creterion, data_loader:Dataloader, logger:Logger) -> Tuple[float, float]:
  tot, ok, loss = 0, 0, 0.0

  model.eval()
  for X_np, Y_np in data_loader():
    X, Y = to_tensor(X_np, Y_np)
  
    logits = model(X)
    l = creterion(logits, Y)
    pred = argmax(logits)

    ok  += (Y == pred).sum()
    tot += len(Y)
    loss += l.item()
  
  return loss / tot, ok / tot

def train(args, model:Module, optimizer, creterion, train_loader:Dataloader, test_loader:Dataloader, logger:Logger) -> LossesAccs:
  step = 0

  losses, accs = [], []
  test_losses, test_accs = [], []
  tot, ok, loss = 0, 0, 0.0
  for e in range(args.epochs):
    logger.info(f'[Epoch {e}/{args.epochs}]')

    model.train()
    for X_np, Y_np in train_loader():
      X, Y = to_tensor(X_np, Y_np)

      optimizer.zero_grad()
      logits = model(X)
      l = creterion(logits, Y)
      l.backward()
      optimizer.step()

      pred = argmax(logits)
      ok  += (Y == pred).sum()
      tot += len(Y)
      loss += l.item()

      step += 1

      if step % 50 == 0:
        losses.append(loss / tot)
        accs  .append(ok   / tot)
        logger.info(f'>> [Step {step}] loss: {losses[-1]}, acc: {accs[-1]:.3%}')
        tot, ok, loss = 0, 0, 0.0

        model.eval()
        tloss, tacc = valid(args, model, creterion, test_loader, logger)
        test_losses.append(tloss)
        test_accs  .append(tacc)
        model.train()

  return losses, accs, test_losses, test_accs


def go_train(args):
  # configs
  args.expname = f'{args.analyzer}_{args.model}'
  out_dp: Path = LOG_PATH / args.analyzer / args.model
  out_dp.mkdir(exist_ok=True, parents=True)
  args.out_dp = str(out_dp)
  logger = get_logger(out_dp / 'run.log', mode='w')

  # symbols (codebook)
  def parse_analyzer(analyzer:str) -> List[str]:
    analyzers = []
    if analyzer.endswith('+'):
      analyzers.append('char')
      analyzer = analyzer[:-1]
    analyzers.append(analyzer)
    return analyzers
  def unify_vocab(analyzers:List[str], min_freq:int=3) -> Vocab:
    vocab_uni = {}
    for analyzer in analyzers:
      vocab_uni.update(load_vocab(LOG_PATH / analyzer / 'vocab.txt'))
    if min_freq: vocab_uni = truncate_vocab(vocab_uni, min_freq)
    return vocab_uni

  analyzers = parse_analyzer(args.analyzer)
  vocab = unify_vocab(analyzers, args.min_freq)
  args.n_vocab = len(vocab) + 1  # <PAD>

  # data
  train_loader = gen_dataloader(args, 'train', vocab)
  test_loader  = gen_dataloader(args, 'test',  vocab)
  valid_loader = gen_dataloader(args, 'valid', vocab)

  # model & optimizer & loss
  model = get_model(args)
  args.param_cnt = sum([p.numel() for p in model.parameters() if p.requires_grad])
  
  optimizer = Adam(model.parameters(), args.lr)
  creterion = CrossEntropyLoss()    # creterion accepts integer label as truth

  # info
  logger.info(f'hparam: {vars(args)}')

  # train
  losses_and_accs = train(args, model, optimizer, creterion, train_loader, test_loader, logger)

  # plot
  plot_loss_and_acc(losses_and_accs, out_dp / PLOT_FILE, title=args.expname)

  # save & load
  save_ckpt(model, out_dp / MODEL_FILE)
  load_ckpt(model, out_dp / MODEL_FILE)
  
  # eval
  result = { 'hparam': vars(args), 'scores': {} }
  for split in SPLITS:
    datat_loader = locals().get(f'{split}_loader')
    precs, recalls, f1s, cmats = [e.tolist() for e in test(args, model, datat_loader, logger)]
    result['scores'][split] = {
      'prec':   precs,
      'recall': recalls,
      'f1':     f1s,
      'cmat':   cmats,
    }
  json_dump(result, out_dp / TASK_FILE)


if __name__ == '__main__':
  args = get_args()

  assert args.model.startswith('cnn'), 'torch impl is only for TextCNN, use `run_baseline_vq.py` to run other models'
  assert not args.eval, 'not supported, use `run_baseline_vq.py` instead'

  go_train(args)
