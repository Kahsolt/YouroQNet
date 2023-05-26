#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/09 

import random
from pprint import pformat
from argparse import ArgumentParser
import warnings ; warnings.simplefilter("ignore")

if 'pyvqnet & pyqpanda':
  # qvm & gates
  from pyqpanda import QCircuit, QProg
  from pyqpanda import X, Y, Z, I, H, S, T
  from pyqpanda import CNOT, CZ, SWAP, iSWAP, SqiSWAP, Toffoli
  from pyqpanda import RX, RY, RZ, P, U1, U2, U3, U4
  from pyqpanda import CR, CU, RXX, RYY, RZZ, RZX
  from pyqpanda import BARRIER
  from pyqpanda import draw_qprog, draw_qprog_text
  # basic
  from pyvqnet import tensor
  from pyvqnet.tensor import QTensor
  # nn
  from pyvqnet.nn.linear import Linear
  from pyvqnet.nn.activation import ReLu, LeakyReLu
  from pyvqnet.nn.dropout import Dropout
  # qnn
  from pyvqnet.qnn.quantumlayer import QuantumLayer, QuantumLayerWithQProg, QuantumLayerMultiProcess, QuantumLayerV2, grad
  # optimizing
  from pyvqnet.qnn.measure import expval, ProbsMeasure, QuantumMeasure, DensityMatrixFromQstate, VN_Entropy, Mutal_Info, Hermitian_expval, MeasurePauliSum, VarMeasure, Purity
  from pyvqnet.nn.loss import BinaryCrossEntropy, SoftmaxCrossEntropy, CategoricalCrossEntropy, CrossEntropyLoss, NLL_Loss, fidelityLoss
  from pyvqnet.optim import SGD, Adam
  from pyvqnet.qnn.opt import SPSA, QNG
  # ckpt
  from pyvqnet.utils.storage import load_parameters
  # randseed
  from pyvqnet.utils import set_random_seed, get_random_seed

import numpy as np

from utils import *
from mk_vocab import make_tokenizer, load_vocab, truncate_vocab, Vocab, VocabI, Tokenizer, PreprocessPack

if 'gate const & utils':
  # constant gates
  S_CGATES = [X, Y, Z, I, H, S, T]
  D_CGATES = [CNOT, CZ, SWAP, iSWAP, SqiSWAP]
  T_CGATES = [Toffoli]
  CGATES   = S_CGATES + D_CGATES + T_CGATES
  # variational gates
  S_PGATES = [RX, RY, RZ, P, U1, U2, U3, U4]
  D_PGATES = [CR, CU, RXX, RYY, RZZ, RZX]
  PGATES   = S_PGATES + D_PGATES

  def get_gate_param(gate:Union[QGate, str]) -> int:
    ''' get the gate param count '''
    if gate in CGATES: return 0
    if gate in [RX, RY, RZ, P, U1, CR, RXX, RYY, RZZ, RZX]: return 1
    if gate in [U2]: return 2
    if gate in [U3]: return 3
    if gate in [U4, CU]: return 4
    if isinstance(gate, str):
      assert gate.startswith('C')
      return get_gate_param(globals()[gate[1:]])
    raise ValueError(gate)

  def is_gate_ctrl(gate:Union[QGate, str]) -> bool:
    ''' is the gate controllable '''
    if isinstance(gate, str):
      return gate.startswith('C')
    else:
      return gate in D_CGATES + T_CGATES + D_PGATES
  
  def gates_to_names(gates:List[QGate]) -> List[str]:
    ''' assert info print '''
    return [g if isinstance(g, str) else g.__name__ for g in gates]
  
  def names_to_gates(names: Union[List[str], str]) -> List[QGate]:
    ''' parse cmdline args '''
    if isinstance(names, str): names = names.split(',')
    return [globals().get(n, n) for n in names]

  VALID_SEC_ROTS  = S_PGATES
  VALID_SEC_ENTGL = D_CGATES + D_PGATES + [f'C{g.__name__}' for g in S_PGATES]
  VALID_CMC_ROTS  = S_PGATES

if 'globals':
  export_circuit = True
  param_frozen = None


def StrongEntangleCircuitTemplate(qv:Qubits, param:NDArray, rots:List[QGate]=[RY, RZ], entgl:Union[QGate, str]=CNOT) -> QCircuit:
  ''' rotate each qubit in register `qv` by `rots` respectively, then entangle them with `entgl` cyclically '''
  
  # sanity check
  assert all([rot in VALID_SEC_ROTS for rot in rots]), f'rots must chosen from {gates_to_names(VALID_SEC_ROTS)}'
  if isinstance(entgl, str):
    assert entgl.startswith('C'), 'engtl must name startswith "C" when passed as "str" type'
    entgl: QGate = globals().get(entgl[1:])
    assert entgl in S_PGATES, f'engtl must be chosen from {[f"C{n}" for n in gates_to_names(S_PGATES)]} when passed as "str" type'
    entgl_has_control = False
  else:
    assert entgl in VALID_SEC_ENTGL, f'engtl must be chosen from {gates_to_names(VALID_SEC_ENTGL)}'
    entgl_has_control = True

  # split param, make *args
  nq = len(qv)
  param = param.reshape(nq, -1)
  cps_r = [0] + np.cumsum([get_gate_param(rot) for rot in rots]).tolist()
  cp    = sum(cps_r)
  param_r = param[:, :cp]
  param_e = param[:, cp:]
  args_r  = [[tuple(args[cps_r[i]:cps_r[i+1]]) for i in range(len(cps_r)-1)] for args in param_r]
  args_e  = [tuple([args] if len(args) else []) for args in param_e]

  qc = QCircuit()
  # rotations
  for i in range(nq):
    for j, rot in enumerate(rots):
      qc << rot(qv[i], *args_r[i][j])
  # entangles
  if entgl_has_control:
    for i in range(nq - 1):
      qc << entgl(qv[i], qv[i + 1], *args_e[i])
    if nq >= 3: qc << entgl(qv[i + 1], qv[0], *args_e[i+1])   # loopback
  else:
    for i in range(nq - 1):
      qc << entgl(qv[i + 1], *args_e[i]).control(qv[i])
    if nq >= 3: qc << entgl(qv[0], *args_e[i+1]).control(qv[i + 1])   # loopback
  return qc

def ControlMultiCircuitTemplate(q:Qubit, qv:Qubits, param:NDArray, rots:List[QGate]=[RY, RZ]) -> QCircuit:
  ''' controlled-rotate each target qubit in register `qv` respectively, under the control of qubit `q` '''
  
  # sanity check
  assert all([rot in VALID_CMC_ROTS for rot in rots]), f'rots must be chosen from {gates_to_names(VALID_CMC_ROTS)}'

  # split param, make *args
  nq = len(qv)
  param = param.reshape(nq, len(rots))
  cps   = [0] + np.cumsum([get_gate_param(rot) for rot in rots]).tolist()
  args  = [[tuple(args[cps[i]:cps[i+1]]) for i in range(len(cps)-1)] for args in param]
  
  # controlled rotations
  qc = QCircuit()
  for i in range(nq):
    for j, rot in enumerate(rots):
      qc << rot(qv[i], *args[i][j]).control(q)
  return qc

def compute_circuit(cq:QCircuit, qv:Qubits, qvm:QVM, index:Union[list, int, slice]=[0], use_qnn_measure:bool=False) -> Probs:
  ''' set mbit to measure on the first m-qubits '''

  prog = QProg() << cq
  if use_qnn_measure:
    assert isinstance(index, list)
    prob = ProbsMeasure(index, prog, qvm, qv)
  else:
    assert isinstance(index, (int, slice))
    prob = qvm.prob_run_list(prog, qv[index])

  if not 'expval':
    print('rlt:', [expval(qvm, prog, {f"Z{i}": 1}, qv) for i in range(len(qv))])

  global export_circuit
  if export_circuit:
    export_circuit = False
    try:
      fp = TMP_PATH / 'circuit.png'
      print(f'>> save circuit to {fp}')
      draw_qprog(prog, output='pic', filename=str(fp))
    except: pass
    fp = TMP_PATH / 'circuit.txt'
    print(f'>> save circuit to {fp}')
    draw_qprog_text(prog, output_file=str(fp))

  return prob

def prob_project(args, x:QTensor) -> QTensor:
  ''' probability distribution projection '''

  # for mixed model, return as is
  if args.mnet: return x
  # for not quantum onehot out, return as is
  if not args.onehot: return x

  # for multi-clf, accumulate joint-distro to marginal-distro [B, D=16=2^4] => [B, D=4=NC]
  B, D = x.shape
  NC = int(np.log2(D))
  if args.proj == 'part_meas':
    return x
  if args.proj == 'onehot':
    return tensor.stack([x[:, 2**k] for k in range(NC)], axis=1)     # NOTE: this sometime not work
  if args.proj == 'trunc':
    return x[:, :NC]


def get_YouroQNet(args) -> QModelInit:
  ''' 熔炉ネットと言うのは、虚仮威し全て裏技を繋ぐもん '''

  if 'hparam':
    # qubits: |p> for context, |q> for data buffer
    if   args.binary: n_qubit_p = 1                                     # use 1-qubit for binary clf
    elif args.onehot: n_qubit_p = args.n_class                          # use nc-qubit for quantum onehot out
    else:             n_qubit_p = int(np.ceil(np.log2(args.n_class)))   # use sqrt(nc)-qubit otherwise
    n_qubit_q = args.n_len
    n_qubit   = n_qubit_p + n_qubit_q

    # circuit
    n_repeat  = args.n_repeat
    SEC_rots  = names_to_gates(args.SEC_rots)
    SEC_entgl = names_to_gates(args.SEC_entgl)[0]
    CMC_rots  = names_to_gates(args.CMC_rots)
    StrongEntangleCircuit = lambda *args, **kwargs: StrongEntangleCircuitTemplate(*args, **kwargs, rots=SEC_rots, entgl=SEC_entgl)
    ControlMultiCircuit   = lambda *args, **kwargs: ControlMultiCircuitTemplate  (*args, **kwargs, rots=CMC_rots)
    
    # params: theta for embed, psi for ansatz
    n_param_SEC = sum(get_gate_param(g) for g in SEC_rots + [SEC_entgl])
    n_param_CMC = sum(get_gate_param(g) for g in CMC_rots)
    n_param_tht_parts = [
      args.n_vocab * n_repeat * n_param_SEC,      # embed
    ]
    n_param_psi_parts = [
      n_qubit_p             * (n_repeat + 1) * n_param_SEC,   # tranx
      n_qubit_q * n_qubit_p *  n_repeat      * n_param_CMC,   # write
      n_qubit_p * n_qubit_q *  n_repeat      * n_param_CMC,   # read
    ]
    n_param_tht = sum(n_param_tht_parts)
    n_param_psi = sum(n_param_psi_parts)
    n_param = n_param_tht + n_param_psi

    if 'add hparam to args':
      args.n_qubit_p   = n_qubit_p
      args.n_qubit_q   = n_qubit_q
      args.n_qubit     = n_qubit
      args.n_repeat    = n_repeat
      args.n_param_tht = n_param_tht
      args.n_param_psi = n_param_psi
      args.n_param     = n_param
      args.embed_dim   = n_repeat * n_param_SEC

  def YouroQNet_qdrl(data:NDArray, param:NDArray, qv:Qubits, cv:Cbits, qvm:QVM):
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
      cp1, cp2, _ = np.cumsum(n_param_psi_parts)
      psi_T = psi[   :cp1] ; psi_T = psi_T.reshape(n_repeat + 1, -1)  # [n_repeat+1=5, 4]
      psi_W = psi[cp1:cp2] ; psi_W = psi_W.reshape(n_repeat,     -1)  # [n_repeat=4, 32]
      psi_R = psi[cp2:]    ; psi_R = psi_R.reshape(n_repeat,     -1)  # [n_repeat=4, 32]
      # rearange theta
      theta = theta.reshape(len(buf), n_repeat, -1)   # [L=16, n_repeat=4, n_gate_param=2]

      # build circuit
      qc = QCircuit()
      if args.init_H:   qc << H(qv)
      if args.init_H_p: qc << H(ctx)
      if args.init_H_q: qc << H(buf)
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
    # alternative freeze
    param_tht = param
    param_psi = param
    if args.alt:
      global param_frozen
      if param_frozen is None:
        param_frozen = param
      else:
        if args.alt_tgt:      # NOTE: True for freeze psi
          param_psi = param_frozen
        else:
          param_tht = param_frozen
        args.alt_cnt -= 1
        if args.alt_cnt <= 0:
          args.alt_cnt = args.alt_step * args.batch_size
          args.alt_tgt = not args.alt_tgt
          param_frozen = param
    # param perturbation for stableness
    if args.noise > 0:
      noise = np.random.normal(size=param.shape) * args.noise
      param_tht += noise
      param_psi += noise
    # split param
    embed = get_embed(args, param_tht)        # whole embed table, [K, D=n_repeat*n_gate_param]
    theta = embed_lookup(args, embed, data)   # sentence related entries, [mL, D]
    psi   = param_psi[n_param_tht:]           # ansatz params
    # build & run circuit
    qc = build_circuit(theta, psi)
    if args.onehot and args.proj == 'part_meas':
      return [compute_circuit(qc, qv, qvm, index=idx)[0] for idx in range(n_qubit_p)]
    else:
      return compute_circuit(qc, qv, qvm, index=slice(None, n_qubit_p))

  return BinaryCrossEntropy, YouroQNet_qdrl, n_qubit, n_param

get_YouroMNet = get_YouroQNet

class MLP(Module):

  def __init__(self, args, qnn:QModel):
    super().__init__()

    self.args = args
    self.qnn = qnn

    assert hasattr(args, 'n_qubit_p'), 'missing args.n_qubit_p'
    self.fc1 = Linear(2**args.n_qubit_p, 2*args.n_qubit_p)
    self.act = ReLu()
    self.drp = Dropout(0.35)
    self.fc2 = Linear(2*args.n_qubit_p, args.n_class)
    args.n_param_linear = sum([p.size for p in self.fc1.parameters() if p.requires_grad]) + \
                          sum([p.size for p in self.fc2.parameters() if p.requires_grad])
    
  def forward(self, x:QTensor):
    z = self.qnn(x)
    o = self.fc1(z)
    o = self.act(o)
    o = self.drp(o)
    o = self.fc2(o)
    return o


def get_model_and_criterion(args) -> Tuple[QModel, Callable]:
  loss_cls, compute_circuit, n_qubit, n_param = globals()[f'get_{args.model}Net'](args)
  model = QuantumLayer(compute_circuit, n_param, 'cpu', n_qubit, 0, GRAD_METH[args.grad_meth], args.grad_dx)
  model.m_para.fill_rand_normal_(m=args.embed_avg, s=args.embed_var)
  if not 'use uniform inits, they do not work in most cases :(':
    model.m_para.fill_rand_uniform_(v=1)
    model.m_para.fill_rand_signed_uniform_(v=1)
    model.m_para.fill_rand_uniform_with_bound_(-np.pi/2, np.pi/2)
  
  args.mnet = args.model.endswith('M')
  if args.mnet: return MLP(args, model), SoftmaxCrossEntropy()
  else:         return model, loss_cls()

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

def get_word_mappings(args, symbols:List[str]) -> Tuple[VocabI, VocabI]:
  if args.pad is not None: syms = [args.pad] + symbols
  syms.sort()
  word2id = { v: i for i, v in enumerate(syms) }
  id2word = { i: v for i, v in enumerate(syms) }
  return word2id, id2word

def get_preprocessor_pack(args, vocab:Vocab) -> PreprocessPack:
  tokenizer = list if args.analyzer == 'char' else make_tokenizer(vocab)
  word2id, id2word = get_word_mappings(args, list(vocab.keys()))
  PAD_ID = word2id.get(args.pad, -1)
  aligner = lambda x: align_ids(x, args.n_len, PAD_ID)
  return tokenizer, aligner, word2id, id2word, PAD_ID

def gen_dataloader(args, dataset:Dataset, vocab:Vocab, shuffle:bool=False) -> Dataloader:
  preproc_pack = get_preprocessor_pack(args, vocab)

  T, Y = dataset
  N = args.limit if args.limit > 0 else len(Y)
  indexes = list(range(N))
  if shuffle: random.shuffle(indexes)

  def iter_by_batch() -> Tuple[NDArray, NDArray]:
    nonlocal args, T, Y, preproc_pack, shuffle

    for i in range(0, N, args.batch_size):
      T_batch, Y_batch = [], []
      for j in range(args.batch_size):
        if i + j >= N: break

        idx = indexes[i + j]
        lbl = np.int32(Y[idx] == args.tgt_cls) if args.binary else Y[idx]
        ids = sent_to_ids (T[idx], preproc_pack)
        tgt = id_to_onehot(lbl, args.n_class)

        if args.debug_step:
          print('txt/lbl:', ids_to_sent(ids, preproc_pack), '/', lbl)
          print('ids:', ids)

        T_batch.append(ids)
        Y_batch.append(tgt)

      if len(T_batch) == args.batch_size:
        yield [np.stack(e, axis=0).astype(np.int32) for e in [T_batch, Y_batch]]
  
    if shuffle and args.limit < 0: random.shuffle(indexes)

  return iter_by_batch    # return a DataLoader generator

def get_ansatz(args, param:NDArray) -> NDArray:
  ''' split out the psi part from the param := (theta, psi) bundle '''
  assert hasattr(args, 'n_param_tht'), f'args.n_param_tht not found, forgot to load from {TASK_FILE} ??'
  psi = param[args.n_param_tht:]
  return psi

def get_embed(args, param:NDArray) -> NDArray:
  ''' split out the theta part from the param := (theta, psi) bundle '''
  assert hasattr(args, 'n_param_tht'), f'args.n_param_tht not found, forgot to load from {TASK_FILE} ??'
  tht = param[:args.n_param_tht]
  return tht.reshape(args.n_vocab, -1)   # [K, D=n_repeat*n_gate_param]

def embed_lookup(args, embed:NDArray, ids:NDArray) -> NDArray:
  ids = ids.astype(np.int32)      # [L], one sample
  theta = embed[ids]              # [mL, D], sentence related embeddings
  return embed_norm(args, theta)

def embed_norm(args, x:NDArray) -> NDArray:
  ''' value norm raw embeddings, assure rotation angle well-ranged '''
  if not args.embed_norm: return x
  return (2 * np.arctan(x)) * args.embed_norm        # [-k*pi, k*pi]


def test(args, model:QModel, criterion, test_loader:Dataloader, logger:Logger) -> Metrics:
  Y_true, Y_pred = [], []
  loss = 0.0

  model.eval()
  for X_np, Y_np in test_loader():
    X, Y = to_qtensor(X_np, Y_np)
  
    joint_probs = model(X)
    probs = prob_project(args, joint_probs)
    l = criterion(Y, probs)
    pred = argmax(probs)

    Y_true.extend(   Y.to_numpy().argmax(-1).tolist())
    Y_pred.extend(pred.to_numpy().tolist())
    loss += l.item()

  Y_true = np.asarray(Y_true, dtype=np.int32)
  Y_pred = np.asarray(Y_pred, dtype=np.int32)
  print('Y_true:', Y_true)
  print('Y_pred:', Y_pred)

  acc, f1 = get_acc_f1(Y_pred, Y_true, args.n_class)
  logger.info(f'>> acc: {acc:.3%}')
  logger.info(f'>> f1: {f1}')
  logger.info(f'>> score: {sum(f1) / len(f1) * 60}')
  return loss / len(Y_true), acc, f1

def train(args, model:QModel, optimizer, criterion, train_loader:Dataloader, test_loader:Dataloader, logger:Logger) -> LossesAccs:
  out_dp = Path(args.out_dp)
  step = 0
  best_f1 = -1

  losses, accs = [], []
  test_losses, test_accs, test_f1s = [], [], []
  tot, ok, loss = 0, 0, 0.0
  for e in range(args.epochs):
    logger.info(f'[Epoch {e}/{args.epochs}]')

    model.train()
    for X_np, Y_np in train_loader():
      Y_lbl = Y_np.argmax(-1)
      X, Y = to_qtensor(X_np, Y_np)

      optimizer.zero_grad()
      joint_probs = model(X)
      probs = prob_project(args, joint_probs)
      l = criterion(Y, probs)
      l.backward()
      optimizer._step()

      pred = to_pred(probs)

      if args.debug_step:
        if args.onehot: print('joint_probs:', joint_probs)
        print('probs:', probs)
        print('true:', Y_lbl)
        print('pred:', pred)

      ok  += (Y_lbl == pred).sum()
      tot += len(Y_np) 
      loss += l.item()

      step += 1

      if step % args.slog_interval == 0:
        logger.info(f'>> [Step {step}] loss: {loss / tot}, acc: {ok / tot:.3%}')
      
      if step % args.log_interval == 0:
        losses.append(loss / tot)
        accs  .append(ok   / tot)
        logger.info(f'>> [Step {step}] loss: {losses[-1]}, acc: {accs[-1]:.3%}')
        tot, ok, loss = 0, 0, 0.0

        model.eval()
        tloss, tacc, tf1 = test(args, model, criterion, test_loader, logger)
        test_losses.append(tloss)
        test_accs  .append(tacc)
        test_f1s   .append(tf1)
        model.train()

        new_f1 = mean(tf1)
        if new_f1 > best_f1:
          best_f1 = new_f1
          logger.info(f'>> better f1 {new_f1} found, save ckpt')
          save_ckpt(model, out_dp / (MODEL_FILE_FMT % 'best'))

      if step % args.ckpt_interval == 0:
        save_ckpt(model, out_dp / (MODEL_FILE_FMT % step))
        plot_loss_and_acc([losses, accs, test_losses, test_accs], out_dp / (PLOT_FILE_FMT % step), title=args.expname)

  return losses, accs, test_losses, test_accs

def infer(args, model:QModel, sent:str, tokenizer:Tokenizer, word2id:Vocab) -> Votes:
  PAD_ID = word2id.get(args.pad, -1)
  ids = [word2id[w] for w in tokenizer(clean_text(sent)) if w in word2id]    # [L]
  if len(ids) < args.n_len:
    ids = align_ids(ids, args.n_len, PAD_ID)

  possible_sp = list(range(len(ids) - args.n_len + 1))
  choose_sp = possible_sp if len(possible_sp) <= args.n_vote else random.sample(possible_sp, args.n_vote)
  X_np = np.stack([ids[sp:sp+args.n_len] for sp in choose_sp], axis=0)   # [V, mL=16]

  X = to_qtensor(X_np)    # [V, mL]
  joint_probs = model(X)  # [V, 2^NC]
  probs = prob_project(args, joint_probs)
  pred = to_pred(probs)   # [V]
  votes = pred.tolist()
  return votes


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
  json_dump({ 'hparam': vars(args) }, out_dp / TASK_FILE)

  # train
  losses_and_accs = train(args, model, optimizer, criterion, train_loader, test_loader, logger)
  if args.mnet: params = model.qnn.m_para.to_numpy()
  else:         params = model    .m_para.to_numpy()

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


def go_infer(args, texts:List[str]=None, name_suffix:str='') -> Union[Preds, Inferer]:
  # configs
  out_dp: Path = LOG_PATH / args.analyzer / f'{args.model}{name_suffix}'
  model_fp = out_dp / (MODEL_FILE_FMT % 'best')
  assert model_fp.exists(), f'you must train this model ({model_fp}) before you can infer from :('
  
  # hparam load
  hparam = json_load(out_dp / TASK_FILE)['hparam']
  for k, v in hparam.items():
    setattr(args, k, v)
  args.seed = hparam['seed']

  # fix seed again
  random.seed    (args.seed)
  np.random.seed (args.seed)
  set_random_seed(args.seed)
  print(f'>> refix rand_seed to {args.seed} :)')

  # ignore training settings
  global export_circuit
  export_circuit = False
  args.noise = 0.0
  args.alt = None

  # symbols (codebook) & preprocessor
  vocab = get_vocab(args)
  tokenizer = list if args.analyzer == 'char' else make_tokenizer(vocab)
  word2id, _ = get_word_mappings(args, list(vocab.keys()))

  # model
  model, _ = get_model_and_criterion(args)
  load_ckpt(model, model_fp)
  
  # make a inferer callable if no text given
  if texts is None:
    return lambda sent: infer(args, model, sent, tokenizer, word2id)

  # predict directly if text is given
  return [mode(infer(args, model, sent, tokenizer, word2id)) for sent in texts]


def go_inspect(args, name_suffix:str='', words:List[str]=None):
  # configs
  out_dp: Path = LOG_PATH / args.analyzer / f'{args.model}{name_suffix}'
  assert out_dp.exists(), 'you must train this model before you can infer from :('

  # hparam load
  hparam = json_load(out_dp / TASK_FILE)['hparam']
  for k, v in hparam.items():
    if not hasattr(args, k):
      setattr(args, k, v)

  # embed load
  ckpt = load_parameters(out_dp / MODEL_FILE)
  if args.mnet: param = ckpt['qnn.m_para'].to_numpy()
  else:         param = ckpt    ['m_para'].to_numpy()
  print('param.shape:', param.shape)
  embed = get_embed(args, param)
  K, D = embed.shape
  print('embed.shape:', embed.shape)
  embed = embed[:10, :]     # NOTE: show only first some entries

  # plot
  embed_n  = embed_norm(args, embed)
  embed_np = np.abs(embed_n)
  k = args.embed_norm
  vn = k * np.pi
  plt.clf()
  plt.subplot(131) ; plt.imshow(embed,    cmap='bwr', vmax=vn, vmin=-vn) ; plt.colorbar() ; plt.title('embed')
  if words: plt.yticks(range(len(words)), words)
  plt.subplot(132) ; plt.imshow(embed_n,  cmap='bwr', vmax=vn, vmin=-vn) ; plt.colorbar() ; plt.title(f'{k*2}*arctan(embed)')
  if words: plt.yticks(range(len(words)), words)
  plt.subplot(133) ; plt.imshow(embed_np, cmap='bwr', vmax=vn, vmin=-vn) ; plt.colorbar() ; plt.title(f'|{k*2}*arctan(embed)|')
  if words: plt.yticks(range(len(words)), words)
  plt.suptitle(f'embed: n_vocab={K} n_dim={D}')
  savefig(out_dp / 'embed.png')


def get_parser():
  MODELS = [name[len('get_'):-len('Net')] for name in globals() if name.startswith('get_') and name.endswith('Net')]

  parser = ArgumentParser()
  # preprocess
  parser.add_argument('-L', '--analyzer', default='kgram+', choices=ANALYZERS, help='tokenize level')
  parser.add_argument('-P', '--pad',      default='\x00',       help='model input pad')
  parser.add_argument('--min_freq',       default=10, type=int, help='min_freq for final embedding vocab')
  parser.add_argument('--limit',          default=-1, type=int, help='limit train data samples')
  # model
  parser.add_argument('-M', '--model', default='YouroQ',  choices=MODELS, help='model name')
  parser.add_argument('--n_len',       default=8,         type=int,       help='model input length (in tokens), aka. n_qubit_q')
  parser.add_argument('--n_class',     default=N_CLASS,   type=int,       help='num of class, related to n_qubit_p')
  parser.add_argument('--n_repeat',    default=1,         type=int,       help='circuit n_repeat, effecting embed depth')
  parser.add_argument('--embed_avg',   default=0.0,       type=float,     help='embedding params normal init mean')
  parser.add_argument('--embed_var',   default=0.2,       type=float,     help='embedding params normal init variance')
  parser.add_argument('--embed_norm',  default=1,         type=float,     help='embedding out value normalize, fatcor of pi (1 means [-pi, pi]); set 0 to disable')
  parser.add_argument('--SEC_rots',    default='RY',      help=f'choose multi from {gates_to_names(VALID_SEC_ROTS)}, comma seperate')
  parser.add_argument('--SEC_entgl',   default='CNOT',    help=f'choose one from {gates_to_names(VALID_SEC_ENTGL)}')
  parser.add_argument('--CMC_rots',    default='RY',      help=f'choose multi from {gates_to_names(VALID_CMC_ROTS)}, comma seperate')
  # model (experimental)
  parser.add_argument('--init_H',     action='store_true', help='init the whole ansatz with H, this does NOT work')
  parser.add_argument('--init_H_p',   action='store_true', help='init |p> with H, this should be a MUST!')
  parser.add_argument('--init_H_q',   action='store_true', help='init |q> with H, this does NOT work')
  parser.add_argument('--noise', default=1e-5, type=float, help='add noise on param')
  parser.add_argument('--alt',        action='store_true', help='alternative freeze theta & psi')
  parser.add_argument('--alt_step',  default=10, type=int, help='alt step for --alt')
  # train
  parser.add_argument('-O', '--optim',      default='Adam', choices=['SGD', 'Adam'],  help='optimizer')
  parser.add_argument('-G', '--grad_meth',  default='fd',   choices=GRAD_METH.keys(), help='grad method')
  parser.add_argument(      '--grad_dx',    default=0.01,   type=float, help='step size for finite_diff')
  parser.add_argument('--lr',               default=0.01,   type=float)
  parser.add_argument('-E', '--epochs',     default=4,      type=int)
  parser.add_argument('-B', '--batch_size', default=4,      type=int)
  parser.add_argument('--slog_interval',    default=10,     type=int, help='log loss/acc')
  parser.add_argument('--log_interval',     default=50,     type=int, help='log & test & reset loss/acc')
  parser.add_argument('--ckpt_interval',    default=200,    type=int, help='save ckpt')
  # infer
  parser.add_argument('--n_vote', default=5, type=int, help='max number of voters at inference time')
  # misc
  parser.add_argument('--seed',       default=RAND_SEED, type=int, help='rand seed')
  parser.add_argument('--inspect',    action='store_true',  help='run embed inspect only')
  parser.add_argument('--debug_step', action='store_true',  help='debug output of each training step')
  parser.add_argument('--binary',     action='store_true',  help='force binary clf mode, override --n_class')
  parser.add_argument('--tgt_cls',    default=0,  type=int, help='relabel the tgt_cls as 1, otherwise 0; work with --binary')
  parser.add_argument('--onehot',     action='store_true',  help='force quantum onehot output, override --n_qubit_p')
  parser.add_argument('--proj',       default='onehot',  choices=['part_meas', 'onehot', 'trunc'], help='prob project method, work with --onehot')
  return parser

def get_args(parser=None):
  parser = parser or get_parser()
  args = parser.parse_args()

  # sanity check
  assert not all([args.binary, args.onehot]), '--binary conflicts with --onehot'

  # fix musts
  args.init_H_p = True    # NOTE: fix this for more quick convergence (still slow though)

  # fix randseed
  seed = args.seed
  if seed < 0:
    random.seed(int(time()))
    seed = random.randint(0, 2**15-1)
  random.seed    (seed)
  np.random.seed (seed)
  set_random_seed(seed)
  args.seed = get_random_seed()
  print(f'>> fix rand_seed to {seed} :)')
  
  # overrides & inits
  if args.binary:
    print('>> binary mode: n_class=2, n_qubit_p=1')
    args.n_class = 2
  
  if args.alt:
    print('>> alternative training tht & psi')
    args.alt_cnt = args.alt_step * args.batch_size
    args.alt_tgt = True     # NOTE: True for freeze psi

  return args


if __name__ == '__main__':
  args = get_args()

  if args.inspect:
    go_inspect(args)
    exit(0)

  go_train(args)
  go_inspect(args)
