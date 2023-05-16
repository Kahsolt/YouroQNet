#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/21 

from argparse import ArgumentParser
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from pyqpanda import QProg
from pyqpanda import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, CR, CP
from pyvqnet.tensor import QTensor
from pyvqnet.qnn import QuantumLayer, ProbsMeasure
from pyvqnet.optim import Adam, SGD
from pyvqnet.nn import MeanSquaredError

from utils import TMP_PATH, GRAD_METH, savefig

# NOTE: a toy qdrl_circuit approximating arbitary normalized prob-dist

# make a triple-face coin: equal prob of [00, 01, 10]
#pdf = [0, 1/3, 1/3, 1/3]
#pdf = [1/3, 0, 1/3, 1/3]
#pdf = [1/3, 1/3, 0, 1/3]
#pdf = [1/3, 1/3, 1/3, 0]
# you can also try any other distribution
#pdf = [1/4, 1/2, 1/8, 1/8]
# random distribution
pdf = None

# modify according to circuit arch
n_gate = 5


def make_qdrl_circuit(n_repeat, input, params, qv, cv, qvm):
  '''
    RZ can be replaced by RX
    RY can be replaced by RX
    CP can be replaced by CR, control order can be swapped
  '''

  # add some noise to stablize
  params += np.random.uniform(size=params.shape) * 1e-5

  prog = QProg() << H(qv)
  i = 0
  for _ in range(n_repeat):
    prog \
      << RZ(qv[0], params[i]) \
      << RY(qv[0], params[i+1]) \
      << CP(qv[0], qv[1], params[i+2]) \
      << RZ(qv[1], params[i+3]) \
      << RY(qv[1], params[i+4])
    i += n_gate

  return ProbsMeasure(list(range(len(qv))), prog, qvm, qv)


def train(args):
  # mock data
  global pdf
  if pdf is None:
    pdf = np.random.uniform(size=[2**args.qubit_num])
    pdf /= pdf.sum()
  assert len(pdf) == 2**args.qubit_num

  # data
  X = [[0.0]]    # dummy
  Y = QTensor(pdf).reshape((1, -1))

  # model & optim
  qdrl_circuit = partial(make_qdrl_circuit, args.n_repeat)
  model = QuantumLayer(qdrl_circuit, args.param_num, 'cpu', args.qubit_num, 0, GRAD_METH[args.grad_meth], args.grad_dx)
  if 'custom init':
    params = model.m_para
    n_params = np.cumprod(params.shape).item()
    init_params = np.linspace(0, np.pi, n_params).reshape(params.shape)
    for i in range(n_params):
      params[i] = init_params[i]
  print('param_cnt:', sum([p.size for p in model.parameters()]))
  criterion = MeanSquaredError()
  
  if args.optimizer == 'Adam':
    optimizer = Adam(model.parameters(), lr=args.lr, beta1=0.9, beta2=0.999, epsilon=1e-8, amsgrad=False)
  elif args.optimizer == 'SGD':
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.w, nesterov=False)

  # book keeper
  param_shot = []
  loss_shot  = []

  # train
  model.train()
  for i in range(args.steps):
    output = model(X)
    loss = criterion(Y, output)
    optimizer.zero_grad()
    loss.backward()
    optimizer._step()

    param_shot.append(model.m_para.to_numpy())
    loss_shot .append(loss.item())

    if i % 100 == 0:
      l = loss.item()
      print(f"step: {i}, loss: {l:g}")
      if l < 1e-8: break
  
  # result
  print('-' * 72)
  print('params:', model.m_para)
  print('-' * 72)
  print('real:', pdf)
  print('pred:', model(X)[0])

  # plots
  plt.clf()
  plt.subplot(211)
  plt.plot(loss_shot)
  plt.subplot(212)
  plt.plot(np.log(loss_shot))
  plt.suptitle('loss & log(loss)')
  savefig(TMP_PATH / 'run_quantum_toy_loss.png')

  param_shot = np.stack(param_shot, axis=0)
  plt.clf()
  for i in range(param_shot.shape[-1]):
    plt.plot(param_shot[:, i], label=i)
  plt.suptitle('params')
  plt.legend()
  savefig(TMP_PATH / 'run_quantum_toy_params.png')
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-S', '--steps',     default=7000, type=int)
  parser.add_argument('-R', '--n_repeat',  default=1,    type=int)
  parser.add_argument('-N', '--qubit_num', default=2,    type=int)
  parser.add_argument('-O', '--optimizer', default='Adam', choices=['Adam', 'SGD'])
  parser.add_argument(      '--w',         default=0.0,  type=float, help='SGD momentum')
  parser.add_argument('-G', '--grad_meth', default='ps', choices=GRAD_METH.keys())
  parser.add_argument('-D', '--grad_dx',   default=0.01, type=float)
  parser.add_argument(      '--lr',        default=0.1,  type=float)
  args = parser.parse_args()

  args.param_num = args.n_repeat * n_gate

  train(args)
