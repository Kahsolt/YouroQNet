#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/09 

import json
from argparse import ArgumentParser
from functools import partial
from traceback import print_exc
import warnings ; warnings.simplefilter("ignore")

if 'pyvqnet & pyqpanda':
  # qvm & gates
  from pyqpanda import CPUQVM
  from pyqpanda import *
  # tensor
  from pyvqnet import tensor
  from pyvqnet.tensor import QTensor
  # basics
  from pyvqnet.nn.module import Module
  # qnn
  from pyvqnet.qnn.quantumlayer import QuantumLayer, QuantumLayerWithQProg, QuantumLayerMultiProcess, QuantumLayerV2, grad
  from pyvqnet.qnn.template import BasicEmbeddingCircuit, AmplitudeEmbeddingCircuit, AngleEmbeddingCircuit, IQPEmbeddingCircuits
  from pyvqnet.qnn.template import RotCircuit, CSWAPcircuit, CRotCircuit, QuantumPoolingCircuit, CCZ, Controlled_Hadamard, BasisState
  from pyvqnet.qnn.template import RandomTemplate, StronglyEntanglingTemplate, BasicEntanglerTemplate, SimplifiedTwoDesignTemplate
  from pyvqnet.qnn.pqc import PQCLayer
  from pyvqnet.qnn.qvc import Qvc
  from pyvqnet.qnn.qdrl import vmodel
  from pyvqnet.qnn.qlinear import QLinear
  from pyvqnet.qnn.qcnn import QConv, Quanvolution
  from pyvqnet.qnn.qae import QAElayer
  from pyvqnet.qnn.qembed import Quantum_Embedding
  # optimizing
  from pyvqnet.qnn.measure import expval, ProbsMeasure, QuantumMeasure, DensityMatrixFromQstate, VN_Entropy, Mutal_Info, Hermitian_expval, MeasurePauliSum, VarMeasure, Purity
  from pyvqnet.nn.loss import CategoricalCrossEntropy, BinaryCrossEntropy, SoftmaxCrossEntropy, CrossEntropyLoss
  from pyvqnet.optim import SGD, Adam, Rotosolve
  from pyvqnet.qnn.opt import SPSA, QNG
  # ckpt
  from pyvqnet.utils.storage import load_parameters, save_parameters

import numpy as np
import matplotlib ; matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils import *
from mk_vocab import make_tokenizer, load_vocab, truncate_vocab, Vocab


def get_NaiveQNet() -> ModelConfig:
  pass

def get_YouroQNet() -> ModelConfig:
  n_qubit = 16
  n_rep_data = 2
  n_param = n_qubit * 2

  def YouroQNet_qdrl(data:NDArray, param:NDArray, qv:Qubits, cv:Cbits, qvm:QVM):
    def build_circuit() -> QCircuit:
      pass

    prog = QProg() << build_circuit()
    if 'use_qmeasure':
      prob = ProbsMeasure([0], prog, qvm, qv)
    else:
      prob = qvm.prob_run_list(prog, qv[0])
    return prob

  return YouroQNet_qdrl, n_qubit, n_param


def get_model(args) -> QuantumLayer:
  compute_circuit, n_qubit, n_param =  globals()[f'get_{args.model}QNet']()
  return QuantumLayer(compute_circuit, n_param, 'cpu', n_qubit)

def get_vocab(args) -> Vocab:
  analyzer: str = args.analyzer
  analyzers = []
  if analyzer.endswith('+'):
    analyzers.append('char')
    analyzer = analyzer[:-1]
  analyzers.append(analyzer)

  vocab = {}
  for analyzer in analyzers:
    vocab.update(load_vocab(LOG_PATH / analyzer / 'vocab.txt'))
  if args.min_freq > 0:
    vocab = truncate_vocab(vocab, args.min_freq)
  return vocab

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
    aligner = partial(align_text, args.length)

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


def test(args, model:QuantumLayer, creterion, test_loader:Dataloader, logger:Logger) -> Metrics:
  Y_true, Y_pred = [], []
  loss = 0.0

  model.eval()
  for X_np, Y_np in test_loader():
    X = QTensor(X_np)
    Y = QTensor(Y_np)
  
    logits = model(X)
    l = creterion(Y, logits)
    pred = logits.argmax(-1, False)   # axis, keepdims

    Y_true.extend(   Y.to_numpy().tolist())
    Y_pred.extend(pred.to_numpy().tolist())
    loss += l.item()

  Y_true = np.asarray(Y_true, dtype=np.int32)
  Y_pred = np.asarray(Y_pred, dtype=np.int32)

  acc, f1 = get_acc_f1(Y_pred, Y_true, args.n_class)
  logger.info(f'>> acc: {acc}')
  logger.info(f'>> f1: {f1}')
  logger.info(f'>> score: {sum(f1) / len(f1)}')
  return loss / len(Y_true), acc, f1

def train(args, model:QuantumLayer, optimizer, creterion, train_loader:Dataloader, test_loader:Dataloader, logger:Logger) -> List[List[float]]:
  step = 0

  losses, accs = [], []
  test_losses, test_accs, test_f1s = [], [], []
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
        tloss, tacc, tf1 = test(args, model, creterion, test_loader, logger)
        test_losses.append(tloss)
        test_accs  .append(tacc)
        test_f1s   .append(tf1)
        model.train()

  return losses, accs, test_losses, test_accs


def go_train(args):
  # configs
  args.expname = f'{args.analyzer}_{args.model}'
  out_dp: Path = LOG_PATH / args.analyzer / args.model
  out_dp.mkdir(exist_ok=True, parents=True)
  logger = get_logger(out_dp / 'run.log', mode='w')

  # symbols (codebook)
  vocab = get_vocab(args)
  args.n_vocab = len(vocab) + 1  # <PAD>
  args.n_class = N_CLASS

  # data
  train_loader = gen_dataloader(args, 'train', vocab)
  test_loader  = gen_dataloader(args, 'test',  vocab)
  valid_loader = gen_dataloader(args, 'valid', vocab)

  # model & optimizer & loss
  model = get_model(args)
  args.param_cnt = sum([p.size for p in model.parameters() if p.requires_grad])
  
  optimizer = SGD(model.parameters(), lr=0.5)
  creterion = BinaryCrossEntropy()

  # info
  logger.info(f'hparams: {vars(args)}')

  B = args.batch_size
  D = 8
  X = tensor.ones([B, D])
  y = model(X)
  breakpoint()

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


def go_infer(args, texts:List[str]) -> List[int]:
  # configs
  out_dp: Path = LOG_PATH / args.analyzer / args.model
  assert out_dp.exists()

  # symbols (codebook)
  vocab = get_vocab(args)
  args.n_vocab = len(vocab) + 1  # <PAD>
  args.n_class = N_CLASS
  
  # model
  model = get_model(args)
  load_ckpt(model, out_dp / 'model.pth')


def go_eval(args):
  raise NotImplementedError


def get_args():
  parser = ArgumentParser()
  parser.add_argument('-L', '--analyzer',   default='kgram+', choices=ANALYZERS, help='tokenize level')
  parser.add_argument('-M', '--model',      default='Youro',        help='model config string pattern')
  parser.add_argument('-N', '--length',     default=32,   type=int, help='model input length (in tokens)')
  parser.add_argument('-P', '--pad',        default='\x00',         help='model input pad')
  parser.add_argument('-D', '--embed_dim',  default=32,   type=int, help='model embed depth')
  parser.add_argument('-E', '--epochs',     default=10,   type=int)
  parser.add_argument('-B', '--batch_size', default=32,   type=int)
  parser.add_argument('--lr',               default=0.1,  type=float)
  parser.add_argument('--min_freq',         default=5,    type=int, help='final vocab for embedding')
  parser.add_argument('--eval', action='store_true', help='compare result scores')
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()

  if args.eval:
    go_eval()
    exit(0)

  go_train(args)
