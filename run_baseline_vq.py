#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/05 

import random
import json
from pathlib import Path
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

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from utils import *
from mk_vocab import make_tokenizer, load_vocab, truncate_vocab, Vocab


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
      self.conv1 = Conv2D(args.embed_dim, args.dim//2, (3, 3), (1, 1), 'same')
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

def gen_dataloader(args, split:str, vocab:Vocab) -> Dataloader:
  def make_mappings(symbols:List[str], pad:str=None) -> Tuple[Dict[str, int], Dict[int, str]]:
    if pad is not None: syms = [pad] + symbols
    syms.sort()
    word2id = { v: i for i, v in enumerate(syms) }
    id2word = { i: v for i, v in enumerate(syms) }
    return word2id, id2word

  symbols = sorted(vocab.keys())
  word2id, id2word = make_mappings(symbols, args.pad)
  tokenizer = make_tokenizer(vocab) if 'gram' in args.analyzer else list

  shuffle = split == 'train'
  T, Y = load_dataset(split)

  def iter_by_batch() -> Tuple[QTensor, QTensor]:
    nonlocal args, shuffle, T, Y, tokenizer, word2id

    N = len(Y)
    indexes = list(range(N))
    if shuffle: random.shuffle(indexes) 
    PAD_ID = word2id.get(args.pad, -1)
    aligner = lambda x: align_words(x, args.length, args.pad)

    for i in range(0, N, args.batch_size):
      T_batch, Y_batch = [], []
      for j in range(args.batch_size):
        if i + j >= N: break
        idx = indexes[i + j]
        T_batch.append(np.asarray([word2id.get(w, PAD_ID) for w in aligner(tokenizer(T[idx]))]))
        Y_batch.append(Y[idx])

      if len(T_batch) == args.batch_size:
        yield [np.stack(e, axis=0).astype(np.int32) for e in [T_batch, Y_batch]]
  
  return iter_by_batch    # return a DataLoader generator


def test(args, model:Module, data_loader:Dataloader, logger:Logger) -> Score:
  Y_true, Y_pred = [], []

  model.eval()
  for X_np, Y_np in data_loader():
    X = QTensor(X_np)
    Y = QTensor(Y_np)
  
    logits = model(X)
    pred = logits.argmax([-1], False)   # axis, keepdims

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

def valid(args, model:Module, creterion, data_loader:Dataloader, logger:Logger) -> Tuple[float, float]:
  tot, ok, loss = 0, 0, 0.0

  model.eval()
  for X_np, Y_np in data_loader():
    X = QTensor(X_np)
    Y = QTensor(Y_np)
  
    logits = model(X)
    l = creterion(Y, logits)
    pred = logits.argmax([-1], False)   # axis, keepdims

    ok  += (Y_np == pred.to_numpy().astype(np.int32)).sum()
    tot += len(Y_np) 
    loss += l.item()
  
  return loss / tot, ok / tot

def train(args, model:Module, optimizer, creterion, train_loader:Dataloader, test_loader:Dataloader, logger:Logger) -> List[List[float]]:
  step = 0

  losses, accs = [], []
  test_losses, test_accs = [], []
  tot, ok, loss = 0, 0, 0.0
  for e in range(args.epochs):
    logger.info(f'[Epoch {e}/{args.epochs}]')

    model.train()
    for X_np, Y_np in train_loader():
      X = QTensor(X_np)
      Y = QTensor(Y_np)

      optimizer.zero_grad()
      logits = model(X)
      l = creterion(Y, logits)
      l.backward()
      optimizer._step()

      pred = logits.argmax([-1], False)   # axis, keepdims
      ok  += (Y_np == pred.to_numpy().astype(np.int32)).sum()
      tot += len(Y_np) 
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
  args.n_class = N_CLASS

  # data
  train_loader = gen_dataloader(args, 'train', vocab)
  test_loader  = gen_dataloader(args, 'test',  vocab)
  valid_loader = gen_dataloader(args, 'valid', vocab)

  # model & optimizer & loss
  model = get_model(args)
  args.param_cnt = sum([p.size for p in model.parameters() if p.requires_grad])
  
  optimizer = Adam(model.parameters(), args.lr)
  creterion = CrossEntropyLoss()

  # info
  logger.info(f'hparams: {vars(args)}')

  # train
  losses, accs, tlosses, taccs = train(args, model, optimizer, creterion, train_loader, test_loader, logger)
  plt.clf()
  ax = plt.axes()
  ax.plot( losses, 'dodgerblue', label='train loss')
  ax.plot(tlosses, 'b',          label='test loss')
  ax2 = ax.twinx()
  ax2.plot( accs, 'orangered', label='train acc')
  ax2.plot(taccs, 'r',         label='test acc')
  plt.legend()
  plt.tight_layout()
  plt.suptitle(args.expname)
  plt.savefig(out_dp / 'loss.png', dpi=600)
  
  # save & load
  save_ckpt(model, out_dp / 'model.pth')
  load_ckpt(model, out_dp / 'model.pth')
  
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
  with open(out_dp / 'result.json', 'w', encoding='utf-8') as fh:
    json.dump(result, fh, indent=2, ensure_ascii=False)


def go_eval(args):
  # just ignored beacuse not good results :(
  raise NotImplementedError

  MODEL_PREFXIES = [ 'dnn', 'cnn', 'rnn' ]
  for analyzer in ANALYZERS:
    out_dp_base = LOG_PATH / analyzer
    if not out_dp_base.exists(): continue

    for dp in out_dp_base.iterdir():
      if not dp.is_dir(): continue
      if not any([dp.name.startswith(prefix) for prefix in MODEL_PREFXIES]): continue

      with open(dp / 'result.json', 'w', encoding='utf-8') as fh:
        result = json.load(fh)['scores']


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-L', '--analyzer', choices=ANALYZERS, help='tokenize level')
  parser.add_argument('-M', '--model',                       help='model config string pattern')
  parser.add_argument('-N', '--length',     default=32,   type=int, help='model input length (in tokens)')
  parser.add_argument('-P', '--pad',        default='\x00',         help='model input pad')
  parser.add_argument('-D', '--embed_dim',  default=32,   type=int, help='model embed depth')
  parser.add_argument('-E', '--epochs',     default=10,   type=int)
  parser.add_argument('-B', '--batch_size', default=32,   type=int)
  parser.add_argument('--lr',               default=1e-3, type=float)
  parser.add_argument('--min_freq',         default=5,    type=int, help='final vocab for embedding')
  parser.add_argument('--eval', action='store_true', help='compare result scores')
  args = parser.parse_args()

  if args.eval:
    go_eval()
    exit(0)

  if args.model.startswith('cnn'):
    print('The TextCNN models are causing C-level kernel dump in loss.backward() !!')
    print('Still do not know why, might be bugs of pyvqnet, so just ignore it :(')
    exit(0)

  go_train(args)
