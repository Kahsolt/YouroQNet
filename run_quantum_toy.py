#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/21 

from pyqpanda import QProg
from pyqpanda import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, CR, CP
from pyvqnet.tensor import QTensor
from pyvqnet.qnn import QuantumLayer, ProbsMeasure
from pyvqnet.optim import Adam
from pyvqnet.nn import MeanSquaredError

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


# repeat circuit block n_times
n_repeat = 2
# n_gate in a block
n_gate   = 5

def qdrl_circuit(input, params, qv, cv, qvm):
  '''
    RZ can be replaced by RX
    CP can be replaced by CR, control order can be swapped
  '''

  prog = QProg() << H(qv)
  i = 0
  for _ in range(n_repeat):
    prog \
      << RX(qv[0], params[i]) \
      << RY(qv[0], params[i+1]) \
      << CR(qv[0], qv[1], params[i+2]) \
      << RX(qv[1], params[i+3]) \
      << RY(qv[1], params[i+4])
    i += n_gate

  return ProbsMeasure(list(range(len(qv))), prog, qvm, qv)


def train(steps=7000):
  qubit_num = 2
  param_num = n_repeat * n_gate

  global pdf
  if pdf is None:
    import numpy as np
    pdf = np.random.uniform(size=[2**qubit_num])
    pdf /= pdf.sum()
  assert len(pdf) == 2**qubit_num

  X = [[0.0]]    # dummy
  Y = QTensor(pdf).reshape((1, -1))

  model = QuantumLayer(qdrl_circuit, param_num, 'cpu', qubit_num)
  print('param_cnt:', sum([p.size for p in model.parameters()]))
  optimizer = Adam(model.parameters(), lr=0.1)
  creterion = MeanSquaredError()

  model.train()
  for i in range(steps):
    output = model(X)
    loss = creterion(Y, output)
    optimizer.zero_grad()
    loss.backward()
    print(model.m_para.grad)
    optimizer._step()

    if i % 20 == 0:
      l = loss.item()
      print(f"step: {i}, loss: {l:g}")
      if l < 1e-15: break
  
  print('-' * 72)
  print('params:', model.parameters())
  print('-' * 72)
  print('real:', pdf)
  print('pred:', model(X)[0])


if __name__ == '__main__':
  train()
