#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/21 

import numpy as np
from pyqpanda import CPUQVM, QCircuit, QProg
from pyqpanda import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, CR, CP
from pyvqnet.tensor import QTensor
from pyvqnet.qnn import QuantumLayer, QuantumLayerWithQProg, QuantumLayerMultiProcess, QuantumLayerV2
from pyvqnet.qnn import ProbsMeasure, grad
from pyvqnet.optim import Adam
from pyvqnet.nn import MeanSquaredError


def test_grad():

  def pqctest(param):
    qvm = CPUQVM()
    qvm.init_qvm()
    qv = qvm.qAlloc_many(2)

    qc = QCircuit() \
      << RX(qv[0], param[0]) \
      << RY(qv[1], param[1]) \
      << CNOT(qv[0], qv[1]) \
      << RX(qv[1], param[2])
    prog = QProg() << qc

    return ProbsMeasure([1], prog, qvm, qv)

  # f and x
  f = pqctest
  x = np.asrray([0.1, 0.2, 0.3])
  dx = 0.01

  # grad by param_shift
  g_ps = grad(f, x)
  print(g_ps)

  # grad by finite_diff
  g_fd = (f(x + dx) - f(x)) / dx
  print(g_fd)

  # grad diff
  d_g = g_ps - g_fd
  print(d_g)


if __name__ == '__main__':
  test_grad()
  