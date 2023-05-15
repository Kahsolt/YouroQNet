#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/09 

import json
from pprint import pformat
from argparse import ArgumentParser
from traceback import print_exc
import warnings ; warnings.simplefilter("ignore")

if 'pyvqnet & pyqpanda':
  # qvm & gates
  from pyqpanda import CPUQVM
  from pyqpanda import draw_qprog, draw_qprog_text
  from pyqpanda import QCircuit, QProg, QGate, QOracle
  from pyqpanda import BARRIER
  from pyqpanda import X, Y, Z, I, H, S, T
  from pyqpanda import CNOT, CZ, SWAP, iSWAP, SqiSWAP, Toffoli
  from pyqpanda import RX, RY, RZ, P, U1, U2, U3, U4
  from pyqpanda import CR, CU, RXX, RYY, RZZ, RZX
  S_GATES  = [X, Y, Z, I, H, S, T]
  D_GATES  = [CNOT, CZ, SWAP, iSWAP, SqiSWAP]
  T_GATES  = [Toffoli]
  GATES    = S_GATES + D_GATES + T_GATES
  S_PGATES = [RX, RY, RZ, P, U1, U2, U3, U4]
  D_PGATES = [CR, CU, RXX, RYY, RZZ, RZX]
  PGATES   = S_PGATES + D_PGATES 
  def get_gate_param(gate:QGate) -> int:
    ''' get the gate param count '''
    if gate in GATES: return 0
    if gate in [RX, RY, RZ, P, U1, CR, RXX, RYY, RZZ, RZX]: return 1
    if gate in [U2]: return 2
    if gate in [U3]: return 3
    if gate in [U4, CU]: return 4
    raise ValueError(gate)
  def is_gate_ctrl(gate:QGate) -> bool:
    ''' is the gate controllable '''
    return gate in D_GATES + T_GATES + D_PGATES
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
  from pyvqnet.nn.loss import BinaryCrossEntropy, SoftmaxCrossEntropy, CategoricalCrossEntropy, CrossEntropyLoss, NLL_Loss, fidelityLoss
  from pyvqnet.optim import SGD, Adam, Rotosolve
  from pyvqnet.qnn.opt import SPSA, QNG
  # ckpt
  from pyvqnet.utils.storage import load_parameters, save_parameters

import numpy as np

from utils import *
from mk_vocab import make_tokenizer, load_vocab, truncate_vocab, Vocab, Tokenizer

class LogOnce:
  export_circuit = True


def StrongEntangleCircuitTemplate(qv:Qubits, param:NDArray, rots:List[QGate]=[RY, RZ], entgl:QGate=CNOT) -> QCircuit:
  ''' rotate each qubit in `qv` by `rots` respectively, then entangle them with `entgl` cyclically '''
  
  # sanity check
  assert all([rot in S_PGATES for rot in rots]), 'rot must be a parameterized single-qubit gate'
  #assert entgl in D_GATES + D_PGATES, 'engtl must be a double-qubit gate'
  assert entgl is CNOT, 'engtl only support CNOT so far'
  nq = len(qv)
  param = param.reshape(nq, len(rots))

  qc = QCircuit()
  # rotations
  for i in range(nq):
    for j, rot in enumerate(rots):
      qc << rot(qv[i], param[i][j])
  # entangles
  for i in range(nq - 1):
    qc << entgl(qv[i], qv[i + 1])
  if nq >= 3:
    qc << entgl(qv[i + 1], qv[0])   # loopback
  return qc

def ControlMultiCircuitTemplate(q:Qubit, qv:Qubits, param:NDArray, rots:List[QGate]=[RY, RZ]) -> QCircuit:
  ''' rotate each target qubit in `qv` respectively, under the control of qubit `q` '''
  
  # sanity check
  #assert entgl in D_GATES + D_PGATES, 'engtl must be a double-qubit gate'
  assert all([rot in S_PGATES for rot in rots]), 'rot must be a parameterized single-qubit gate'
  nq = len(qv)
  param = param.reshape(nq, len(rots))

  # controlled rotations
  qc = QCircuit()
  for i in range(nq):
    for j, rot in enumerate(rots):
      qc << rot(qv[i], param[i][j]).control(q)
  return qc

def compute_circuit(cq:QCircuit, qv:Qubits, qvm:QVM, mbit:int=1) -> Probs:
  ''' set mbit to measure on the first m-qubits '''

  prog = QProg() << cq
  if 'use_qmeasure':
    prob = ProbsMeasure(list(range(mbit)), prog, qvm, qv)
  else:
    prob = qvm.prob_run_list(prog, qv[:mbit])

  if not 'expval':
    print('rlt:', [expval(qvm, prog, {f"Z{i}": 1}, qv) for i in range(len(qv))])

  if LogOnce.export_circuit:
    LogOnce.export_circuit = False
    try: draw_qprog(prog, output='pic', filename=str(TMP_PATH/'circuit.png'))
    except: pass
    draw_qprog_text(prog, output_file=str(TMP_PATH/'circuit.txt'))

  return prob


def get_YouroQNet(args) -> QModelInit:
  ''' 熔炉ネットと言うのは、虚仮威し全て裏技を繋ぐもん '''

  if 'hparams':
    # qubits: |p> for context, |q> for data buffer
    n_qubit_p = int(np.ceil(np.log2(args.n_class)))     # FIXME: need at leaset `n_class - 1`` qubits
    n_qubit_q = args.length
    n_qubit   = n_qubit_p + n_qubit_q

    # circuit
    n_repeat  = args.repeat
    SEC_rots  = [RY, RZ]    # can tune this
    SEC_entgl = CNOT        # NOTE: fixed for the moment
    CMC_rots  = [RY, RZ]    # can tune this
    if 'render circuit templates':
      StrongEntangleCircuit = lambda *args, **kwargs: StrongEntangleCircuitTemplate(*args, **kwargs, rots=SEC_rots, entgl=SEC_entgl)
      ControlMultiCircuit   = lambda *args, **kwargs: ControlMultiCircuitTemplate  (*args, **kwargs, rots=CMC_rots)
    
    # params: theta for embed, psi for ansatz
    n_param_SEC = sum(get_gate_param(g) for g in SEC_rots + [SEC_entgl])
    n_param_CMC = sum(get_gate_param(g) for g in CMC_rots)
    n_param_t_parts = [
      args.n_vocab * n_repeat * n_param_SEC,      # embed
    ]
    n_param_p_parts = [
      n_qubit_p             * (n_repeat + 1) * n_param_SEC,   # tranx
      n_qubit_q * n_qubit_p *  n_repeat      * n_param_CMC,   # write
      n_qubit_p * n_qubit_q *  n_repeat      * n_param_CMC,   # read
    ]
    n_param_t = sum(n_param_t_parts)
    n_param_p = sum(n_param_p_parts)
    n_param   = n_param_t + n_param_p

    if 'add hparams to args':
      args.n_qubit_p   = n_qubit_p
      args.n_qubit_q   = n_qubit_q
      args.n_qubit     = n_qubit
      args.n_repeat    = n_repeat
      args.n_param_SEC = n_param_SEC
      args.n_param_CMC = n_param_CMC
      args.n_param_t   = n_param_t
      args.n_param_p   = n_param_p
      args.n_param     = n_param

  def YouroQNet_qdrl(data:NDArray, param:NDArray, qv:Qubits, cv:Cbits, qvm:QVM):
    def embed_lookup(embed:NDArray, data:NDArray) -> NDArray:
      ids = data.astype(np.int32)               # [L], one sample
      embed = embed.reshape(args.n_vocab, -1)   # [K, D=n_repeat*n_gate_param], whole embeding table
      theta = embed[ids]                        # sentence related embeddings
      theta_n = 2 * np.arctan(theta)            # NOTE: assure angle range in [-pi, pi]
      return theta_n

    def build_buf_load(theta:NDArray) -> QCircuit:
      nonlocal buf
      return StrongEntangleCircuit(buf, theta)
    
    def build_ctx_tranx(psi:NDArray) -> QCircuit:
      nonlocal ctx
      return StrongEntangleCircuit(ctx, psi)

    def build_ctx_write(psi:NDArray) -> QCircuit:
      nonlocal ctx, buf
      nq = len(ctx)
      psi = psi.reshape(nq, -1)
      qc = QCircuit()
      for i in range(nq):
        qc << ControlMultiCircuit(ctx[i], buf, psi[i])
      return qc

    def build_ctx_read(psi:NDArray) -> QCircuit:
      nonlocal ctx, buf
      nq = len(buf)
      psi = psi.reshape(nq, -1)
      qc = QCircuit()
      for i in range(nq):
        qc << ControlMultiCircuit(buf[i], ctx, psi[i])
      return qc

    def build_circuit(theta:NDArray, psi:NDArray) -> QCircuit:
      # split psi
      cp1, cp2, _ = np.cumsum(n_param_p_parts)
      psi_T = psi[   :cp1] ; psi_T = psi_T.reshape(n_repeat + 1, -1)  # [n_repeat+1=5, 4]
      psi_W = psi[cp1:cp2] ; psi_W = psi_W.reshape(n_repeat,     -1)  # [n_repeat=4, 32]
      psi_R = psi[cp2:]    ; psi_R = psi_R.reshape(n_repeat,     -1)  # [n_repeat=4, 32]
      # rearange theta
      theta = theta.reshape(len(buf), n_repeat, -1)   # [L=16, n_repeat=4, n_gate_param=2]

      # build circuit
      qc = QCircuit() # << H(qv)
      # NOTE: to keep code syntacical aligned, use the last portion for init
      qc << build_ctx_tranx(psi_T[-1, :]) \
         << BARRIER(qv)
      for i in range(n_repeat):
        qc << build_buf_load (theta[:, i, :]) \
           << BARRIER(qv) \
           << build_ctx_write(psi_W[i, :]) \
           << BARRIER(qv) \
           << build_ctx_tranx(psi_T[i, :]) \
           << BARRIER(qv) \
           << build_ctx_read (psi_R[i, :]) \
           << BARRIER(qv)
      return qc

    # split qubits
    ctx: Qubits = qv[:n_qubit_p][::-1]  # |p>, current context
    buf: Qubits = qv[n_qubit_p:]        # |q>, placeholder for input sequence
    # split param
    embed = param[:n_param_t]           # whole embedding table
    theta = embed_lookup(embed, data)   # sentence related entries
    psi   = param[n_param_t:]           # ansatz params
    return compute_circuit(build_circuit(theta, psi), qv, qvm, n_qubit_p)

  return YouroQNet_qdrl, BinaryCrossEntropy, n_qubit, n_param


def get_model_and_creterion(args) -> Tuple[QModel, Callable]:
  compute_circuit, loss_cls, n_qubit, n_param = globals()[f'get_{args.model}QNet'](args)
  return QuantumLayer(compute_circuit, n_param, 'cpu', n_qubit, 0, GRAD_METH[args.grad_meth], args.grad_dx), loss_cls()

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

def get_word2id(args, symbols:List[str]) -> Vocab:
  if args.pad is not None: syms = [args.pad] + symbols
  syms.sort()
  word2id = { v: i for i, v in enumerate(syms) }
  return word2id

def gen_dataloader(args, dataset:Dataset, vocab:Vocab, shuffle:bool=False) -> Dataloader:
  T, Y = dataset
  word2id = get_word2id(args, list(vocab.keys()))
  tokenizer = make_tokenizer(vocab) if 'gram' in args.analyzer else list

  def iter_by_batch() -> Tuple[NDArray, NDArray]:
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
        Y_batch.append(np.eye(args.n_class)[Y[idx]])    # make onehot

      if len(T_batch) == args.batch_size:
        yield [np.stack(e, axis=0).astype(np.int32) for e in [T_batch, Y_batch]]
  
  return iter_by_batch    # return a DataLoader generator


def test(args, model:QModel, creterion, test_loader:Dataloader, logger:Logger) -> Metrics:
  Y_true, Y_pred = [], []
  loss = 0.0

  model.eval()
  for X_np, Y_np in test_loader():
    X, Y = to_qtensor(X_np, Y_np)
  
    logits = model(X)
    l = creterion(Y, logits)
    pred = argmax(logits)

    Y_true.extend(   Y.to_numpy().argmax(-1).tolist())
    Y_pred.extend(pred.to_numpy().tolist())
    loss += l.item()

  Y_true = np.asarray(Y_true, dtype=np.int32)
  Y_pred = np.asarray(Y_pred, dtype=np.int32)
  print('Y_true', Y_true)
  print('Y_pred', Y_pred)

  acc, f1 = get_acc_f1(Y_pred, Y_true, args.n_class)
  logger.info(f'>> acc: {acc:.3%}')
  logger.info(f'>> f1: {f1}')
  logger.info(f'>> score: {sum(f1) / len(f1) * 60}')
  return loss / len(Y_true), acc, f1

def train(args, model:QModel, optimizer, creterion, train_loader:Dataloader, test_loader:Dataloader, logger:Logger) -> LossesAccs:
  step = 0

  losses, accs = [], []
  test_losses, test_accs, test_f1s = [], [], []
  tot, ok, loss = 0, 0, 0.0
  for e in range(args.epochs):
    #logger.info(f'[Epoch {e}/{args.epochs}]')

    model.train()
    for X_np, Y_np in train_loader():
      X, Y = to_qtensor(X_np, Y_np)

      optimizer.zero_grad()
      logits = model(X)
      l = creterion(Y, logits)
      l.backward()
      optimizer._step()

      pred = argmax(logits)
      ok  += (Y_np.argmax(-1) == pred.to_numpy().astype(np.int32)).sum()
      tot += len(Y_np) 
      loss += l.item()

      step += 1

      if step % args.log_interval == 0:
        logger.info(f'>> [Step {step}] loss: {loss / tot}, acc: {ok / tot:.3%}')
      
      if step % args.log_reset_interval == 0:
        losses.append(loss / tot)
        accs  .append(ok   / tot)
        logger.info(f'>> [Step {step}] loss: {losses[-1]}, acc: {accs[-1]:.3%}')
        tot, ok, loss = 0, 0, 0.0

      if step % args.test_interval == 0:
        model.eval()
        tloss, tacc, tf1 = test(args, model, creterion, test_loader, logger)
        test_losses.append(tloss)
        test_accs  .append(tacc)
        test_f1s   .append(tf1)
        model.train()

  return losses, accs, test_losses, test_accs

def infer(args, model:QModel, sent:str, tokenizer:Tokenizer, word2id:Vocab) -> Votes:
  PAD_ID = word2id.get(args.pad, -1)
  toks = tokenizer(clean_text(sent))
  if len(toks) < args.length:
    toks = align_words(toks, args.length, args.pad)

  ids = np.asarray([word2id.get(w, PAD_ID) for w in toks], dtype=np.int32)  # [L]
  possible_sp = list(range(len(ids) - args.length + 1))
  choose_sp = possible_sp if len(possible_sp) <= args.n_vote else random.sample(possible_sp, args.n_vote)
  X_np = np.stack([ ids[sp:sp+args.length] for sp in choose_sp ], axis=0)   # [V, mL=16]

  X = to_qtensor(X_np)   # [V, mL]
  logits = model(X)     # [V, NC]
  pred = argmax(logits) # [V]
  votes = pred.to_numpy().tolist()
  return votes


def go_train(args):
  # configs
  args.expname = f'{args.analyzer}_{args.model}'
  out_dp: Path = LOG_PATH / args.analyzer / args.model
  out_dp.mkdir(exist_ok=True, parents=True)
  logger = get_logger(out_dp / 'run.log', mode='w')

  # symbols (codebook)
  vocab = get_vocab(args)
  args.n_vocab = len(vocab) + 1  # <PAD>

  # data
  train_loader = gen_dataloader(args, load_dataset('train'), vocab, shuffle=True)
  test_loader  = gen_dataloader(args, load_dataset('test'),  vocab)
  valid_loader = gen_dataloader(args, load_dataset('valid'), vocab)

  # model & optimizer & loss
  model, creterion = get_model_and_creterion(args)    # creterion accepts onehot label as truth
  args.param_cnt = sum([p.size for p in model.parameters() if p.requires_grad])
  
  if args.optim == 'SGD':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
  elif args.optim == 'Adam':
    optimizer = Adam(model.parameters(), lr=args.lr)

  # info
  logger.info(f'hparams: {pformat(vars(args))}')

  # train
  losses_and_accs = train(args, model, optimizer, creterion, train_loader, test_loader, logger)
  
  # plot
  plot_loss_and_acc(losses_and_accs, out_dp / 'loss_acc.png', title=args.expname)
  
  # save & load
  save_ckpt(model, out_dp / 'model.pth')
  load_ckpt(model, out_dp / 'model.pth')
  
  # eval
  result = { 'hparam': vars(args), 'scores': {} }
  for split in SPLITS:
    datat_loader = locals().get(f'{split}_loader')
    loss, acc, f1 = test(args, model, creterion, datat_loader, logger)
    result['scores'][split] = {
      'loss': loss,
      'acc':  acc,
      'f1':   f1,
    }
  with open(out_dp / 'result.json', 'w', encoding='utf-8') as fh:
    json.dump(result, fh, indent=2, ensure_ascii=False)


def go_infer(args, texts:List[str]=None) -> Union[Votes, Inferer]:
  # configs
  out_dp: Path = LOG_PATH / args.analyzer / args.model
  assert out_dp.exists(), 'you must train this model before you can infer from :('
  
  # model
  model, _ = get_model_and_creterion(args)
  load_ckpt(model, out_dp / 'model.pth')

  # symbols (codebook)
  vocab = get_vocab(args)
  word2id = get_word2id(args, list(vocab.keys()))
  tokenizer = make_tokenizer(vocab) if 'gram' in args.analyzer else list
  args.n_vocab = len(vocab) + 1  # <PAD>

  # make a inferer callable if no text given
  if texts is None:
    return lambda sent: infer(args, model, sent, tokenizer, word2id)

  # predict directly if text is given
  preds = []
  n_ties = 0
  for sent in texts:
    votes = infer(args, model, sent, tokenizer, word2id)
    final = mode(votes)
    if votes.count(final) > 1:
      print(f'warn: meets a tie {votes}')
      n_ties += 1
    preds.append(final)
  if n_ties: print(f'n_ties: {n_ties}')
  
  return preds


def go_eval(args):
  raise NotImplementedError


def get_args():
  MODELS = [name[len('get_'):-len('QNet')] for name in globals() if name.startswith('get_') and name.endswith('QNet')]

  parser = ArgumentParser()
  parser.add_argument('-L', '--analyzer',   default='kgram+', choices=ANALYZERS, help='tokenize level')
  parser.add_argument('-M', '--model',      default='Youro',  choices=MODELS,    help='model config string pattern')
  parser.add_argument('-O', '--optim',      default='SGD',    choices=['SGD', 'Adam'],  help='optimizer')
  parser.add_argument('-G', '--grad_meth',  default='fd',     choices=GRAD_METH.keys(), help='grad method')
  parser.add_argument(      '--grad_dx',    default=0.01,     type=float, help='step size for finite_diff')
  parser.add_argument('-P', '--pad',        default='\x00',         help='model input pad')
  parser.add_argument('-N', '--length',     default=3,    type=int, help='model input length (in tokens)')
  parser.add_argument('-D', '--repeat',     default=2,    type=int, help='circuit n_repeat, effecting embed depth')
  parser.add_argument('-E', '--epochs',     default=10,   type=int)
  parser.add_argument('-B', '--batch_size', default=4,    type=int)
  parser.add_argument('--lr',               default=0.1,  type=float)
  parser.add_argument('--min_freq',         default=5,    type=int, help='min_freq for final embedding vocab')
  parser.add_argument('--n_class',    default=N_CLASS,    type=int, help='num of class')
  parser.add_argument('--n_vote',           default=5,    type=int, help='max number of voters at inference time')
  parser.add_argument('--log_interval',       default=10,  type=int)
  parser.add_argument('--log_reset_interval', default=50,  type=int)
  parser.add_argument('--test_interval',      default=200, type=int)
  parser.add_argument('--eval', action='store_true', help='compare result scores')
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()

  if args.eval:
    go_eval(args)
    exit(0)

  go_train(args)
