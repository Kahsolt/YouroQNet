#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/05 

from pathlib import Path
from pprint import pformat
from argparse import ArgumentParser
import warnings ; warnings.simplefilter("ignore")

if 'pyvqnet':
  # tensor
  from pyvqnet import tensor
  from pyvqnet.tensor import QTensor
  # data
  from pyvqnet.data import data_generator
  # basics
  from pyvqnet.nn.parameter import Parameter
  from pyvqnet.nn.module import Module
  from pyvqnet.nn.pixel_shuffle import Pixel_Shuffle, Pixel_Unshuffle
  # linear
  from pyvqnet.nn.embedding import Embedding
  from pyvqnet.nn.linear import Linear
  # conv
  from pyvqnet.nn.conv import Conv2D, Conv1D, ConvT2D
  from pyvqnet.nn.pooling import MaxPool1D, MaxPool2D, AvgPool1D, AvgPool2D
  # recurrent
  from pyvqnet.nn.rnn import RNN, Dynamic_RNN
  from pyvqnet.nn.gru import GRU, Dynamic_GRU
  from pyvqnet.nn.lstm import LSTM, Dynamic_LSTM
  # net structure
  from pyvqnet.nn.self_attention import Self_Conv_Attention
  from pyvqnet.nn.transformer import Transformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, MultiHeadAttention
  # regularity
  from pyvqnet.nn.dropout import Dropout
  from pyvqnet.nn.batch_norm import BatchNorm1d, BatchNorm2d
  from pyvqnet.nn.layer_norm import LayerNorm1d, LayerNorm2d, LayerNormNd
  from pyvqnet.nn.spectral_norm import Spectral_Norm
  # activation
  from pyvqnet.nn.activation import Sigmoid, ReLu, LeakyReLu, Softmax, Softplus, Softsign, HardSigmoid, ELU, Tanh
  # optimizing
  from pyvqnet.nn.loss import CategoricalCrossEntropy, BinaryCrossEntropy, SoftmaxCrossEntropy, CrossEntropyLoss
  from pyvqnet.optim import SGD, Adam
  # ckpt
  from pyvqnet.utils.storage import load_parameters, save_parameters

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from utils import *
from run_quantum import get_vocab, gen_dataloader


class TextModel(Module):

  def __init__(self, args):
    super().__init__()

    self.args = args
    self.embed = Embedding(args.n_vocab, args.embed_dim)  # [K, D]

    if hasattr(args, 'agg'):
      if args.agg == 'avg': self.agg = tensor.mean
      if args.agg == 'max': self.agg = tensor.max

class TextDNN(TextModel):

  def __init__(self, args):
    super().__init__(args)

    self.fc1 = Linear(args.embed_dim, args.dim)
    self.act = ReLu()
    self.drp = Dropout(0.35)
    self.fc2 = Linear(args.dim, args.n_class)

  def forward(self, x:QTensor):
    z = self.embed(x)   # [B, L=32, D=32]
    z = self.agg(z, 1)  # [B, D=32]
    o = self.fc1(z)     # [B, D=80]
    o = self.act(o)
    if self.args.use_res:
      o = o + z
    o = self.drp(o)
    o = self.fc2(o)     # [B, NC=4]
    return o

class TextCNN(TextModel):

  def __init__(self, args):
    super().__init__(args)
  
    if args.type == '1d':
      self.conv1 = Conv1D(args.embed_dim, args.dim//2, 3, 1, 'same')
      self.pool  = MaxPool1D(2, 2)
      self.conv2 = Conv1D(args.dim//2, args.dim, 3, 1, 'same')
    if args.type == '2d':
      self.conv1 = Conv2D(1, args.dim//2, (3, 3), (1, 1), 'same')
      self.pool  = MaxPool2D((2, 2), (2, 2))
      self.conv2 = Conv2D(args.dim//2, args.dim, (3, 3), (1, 1), 'same')

    self.act = ReLu()
    self.drp = Dropout(0.35)
    self.fc  = Linear(args.dim, args.n_class)

  def forward(self, x:QTensor):
    o = self.embed(x)             # [B, L=32, D=32]
    o = tensor.permute(o, [0, 2, 1])  # [B, C=32, L=32]

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
      o = tensor.unsqueeze(o, 1)  # [B,  C=1, H=32, W=32]
      o = self.conv1(o)
      o = self.pool (o)           # [B, C=40, H=16, W=16]
      o = self.act  (o)
      o = self.conv2(o)
      o = self.pool (o)           # [B, C=80, H=8, W=8]
      o = self.act  (o)
      B, C, H, W = o.shape
      o = tensor.reshape(o, (B, C, H * W))
      o = self.agg  (o, -1)       # [B, D=80]
      o = self.drp  (o)
      o = self.fc   (o)           # [B, NC=4]

    return o

class TextRNN(TextModel):

  def __init__(self, args):
    super().__init__(args)

    self.rnn = globals()[args.type](args.embed_dim, args.dim, 1, bidirectional=args.bidir)
    self.act = ReLu()
    self.drp = Dropout(0.35)
    self.fc  = Linear(args.dim * (1 + args.bidir), args.n_class)

  def forward(self, x:QTensor):
    args = self.args

    z = self.embed(x)   # [B, L, D]
    
    if args.type == 'GRU':
      o, h = self.rnn(z)  # [B, L, D*dir]
    elif args.type == 'LSTM':
      B, L, D = z.shape
      h0 = tensor.ones([(1 + args.bidir) * 1, B, args.dim])
      c0 = tensor.ones([(1 + args.bidir) * 1, B, args.dim])
      o, (hn, cn) = self.rnn(z, (h0, c0))
    else: raise ValueError

    if args.agg == 'fin':
      o = o[:, -1, :]     # [B, D*dir], only last frame
    else:
      o = self.agg(o, 1)  # max or avg on dim-1
    o = self.act(o)
    o = self.drp(o)
    o = self.fc (o)     # [B, NC]
    return o


def get_model(args) -> Module:
  name: str = args.model

  if name.startswith('dnn'):
    try:
      _, type, dim, agg = name.split('-')

      dim = int(dim) ; assert dim > 0
      assert type in ['std', 'res']
      if type == 'res': assert dim == args.embed_dim, f'when use residual, must assure dim({dim}) == embed_dim({args.embed_dim})'
      assert agg in ['avg', 'max']
    except:
      print_exc()
      raise ValueError('name pattern should follow: dnn-[type:str]-[dim:int]-[agg:str]')
    
    args.use_res = type == 'res'
    args.dim = dim
    args.agg = agg
    return TextDNN(args)
  
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
  
  if name.startswith('rnn'):
    try:
      _, type, dim, agg, dir = name.split('-')

      dim   = int(dim)   ; assert dim   > 0
      assert type in ['lstm', 'gru']
      assert agg  in ['avg', 'max', 'fin']
      assert dir  in ['uni', 'bi']
    except:
      print_exc()
      raise ValueError('name pattern should follow: rnn-[type:str]-[dim:int]-[agg:str]-[dir:str]')
    
    args.type  = type.upper()
    args.agg   = agg
    args.dim   = dim
    args.bidir = dir == 'bi'
    return TextRNN(args)

  raise ValueError(name)


def test(args, model:Module, data_loader:Dataloader, logger:Logger) -> Score:
  Y_true, Y_pred = [], []

  model.eval()
  for X_np, Y_np in data_loader():
    X, Y = to_qtensor(X_np, Y_np)
  
    logits = model(X)
    pred = argmax(logits)

    Y_true.extend(   Y.to_numpy().tolist())
    Y_pred.extend(pred.to_numpy().tolist())
  
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

def valid(args, model:Module, criterion, data_loader:Dataloader, logger:Logger) -> Tuple[float, float]:
  tot, ok, loss = 0, 0, 0.0

  model.eval()
  for X_np, Y_np in data_loader():
    X, Y = to_qtensor(X_np, Y_np)
  
    logits = model(X)
    l = criterion(Y, logits)
    pred = argmax(logits)

    ok  += (Y_np == pred.to_numpy().astype(np.int32)).sum()
    tot += len(Y_np) 
    loss += l.item()
  
  return loss / tot, ok / tot

def train(args, model:Module, optimizer, criterion, train_loader:Dataloader, test_loader:Dataloader, logger:Logger) -> LossesAccs:
  step = 0

  losses, accs = [], []
  test_losses, test_accs = [], []
  tot, ok, loss = 0, 0, 0.0
  for e in range(args.epochs):
    logger.info(f'[Epoch {e}/{args.epochs}]')

    model.train()
    for X_np, Y_np in train_loader():
      X, Y = to_qtensor(X_np, Y_np)

      optimizer.zero_grad()
      logits = model(X)
      l = criterion(Y, logits)
      l.backward()
      optimizer._step()

      pred = argmax(logits)
      ok  += (Y_np == pred.to_numpy().astype(np.int32)).sum()
      tot += len(Y_np) 
      loss += l.item()

      step += 1

      if step % args.log_interval == 0:
        losses.append(loss / tot)
        accs  .append(ok   / tot)
        logger.info(f'>> [Step {step}] loss: {losses[-1]}, acc: {accs[-1]:.3%}')
        tot, ok, loss = 0, 0, 0.0

      if step % args.test_interval == 0:
        model.eval()
        tloss, tacc = valid(args, model, criterion, test_loader, logger)
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
  vocab = get_vocab(args)
  args.n_vocab = len(vocab) + 1  # <PAD>

  # data
  train_loader = gen_dataloader(args, load_dataset('train'), vocab, shuffle=True)
  test_loader  = gen_dataloader(args, load_dataset('test'),  vocab)
  valid_loader = gen_dataloader(args, load_dataset('valid'), vocab)

  # model & optimizer & loss
  model = get_model(args)
  criterion = CrossEntropyLoss()    # criterion accepts integer label as truth
  args.param_cnt = sum([p.size for p in model.parameters() if p.requires_grad])
  
  if args.optim == 'SGD':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
  elif args.optim == 'Adam':
    optimizer = Adam(model.parameters(), lr=args.lr)

  # info
  logger.info(f'hparam: {pformat(vars(args))}')

  # train
  losses_and_accs = train(args, model, optimizer, criterion, train_loader, test_loader, logger)

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


def get_args():
  parser = ArgumentParser()
  parser.add_argument('-L', '--analyzer',   default='kgram+', choices=ANALYZERS, help='tokenize level')
  parser.add_argument('-M', '--model',      required=True,    help='model config string pattern')
  parser.add_argument('-O', '--optim',      default='SGD',    choices=['SGD', 'Adam'],  help='optimizer')
  parser.add_argument('-N', '--length',     default=32,   type=int, help='model input length (in tokens)')
  parser.add_argument('-P', '--pad',        default='\x00',         help='model input pad')
  parser.add_argument('-D', '--embed_dim',  default=32,   type=int, help='model embed depth')
  parser.add_argument('-E', '--epochs',     default=50,   type=int)
  parser.add_argument('-B', '--batch_size', default=32,   type=int)
  parser.add_argument('--lr',               default=1e-3, type=float)
  parser.add_argument('--min_freq',         default=5,    type=int, help='final vocab for embedding')
  parser.add_argument('--n_class',    default=N_CLASS,    type=int, help='num of class')
  parser.add_argument('--seed',     default=RAND_SEED,    type=int, help='rand seed')
  parser.add_argument('--log_interval',     default=50,   type=int, help='log & reset loss/acc')
  parser.add_argument('--test_interval',    default=200,  type=int, help='test on valid split')
  args = parser.parse_args()

  try_fix_randseed(args.seed)
  return args


if __name__ == '__main__':
  args = get_args()

  if args.model.startswith('cnn'):
    print('The TextCNN models are possibly causing C-level kernel dump in loss.backward() !!')
    print('Do not know why, might be bugs of pyvqnet :(')
    print('=> please use `run_baseline_tc.py` instead :)')

  go_train(args)
