#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/14 

from argparse import ArgumentParser
from typing import Callable

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from pyqpanda import CPUQVM, QCircuit, QProg
from pyqpanda import deep_copy
from pyqpanda import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, CR, CP
from pyvqnet.qnn import ProbsMeasure
from pyvqnet.qnn import RandomTemplate

from utils import TMP_PATH, timer


if 'qvm':
  qvm = CPUQVM()
  qvm.init_qvm()
  qv = qvm.qAlloc_many(2)

  
def qc_simple(param):
  global qvm, qv

  qc = QCircuit() \
    << RX(qv[0], param[0]) \
    << RX(qv[0], np.pi/3)

  return ProbsMeasure([0], QProg() << qc, qvm, qv)

def qc_complex(param):
  global qvm, qv

  qc = QCircuit()
  qc << RX(qv[0], param[0])
  qc << RZ(qv[0], pi/2)
  qc << RY(qv[0], param[1])
  qc << RZ(qv[0], -pi/4)
  qc << CNOT(qv[0], qv[1])
  qc << RZ(qv[1], param[1])
  qc << RX(qv[0], pi/7)
  qc << RY(qv[1], param[0])
  qc << RZ(qv[0], -pi/8)
  qc << CNOT(qv[1], qv[0])
  qc << deep_copy(qc)
  qc << deep_copy(qc)
  qc << deep_copy(qc)
  qc << deep_copy(qc)

  return ProbsMeasure([0], QProg() << qc, qvm, qv)

def qc_rand(param, seed=114514):
  global qvm, qv

  weights = param.reshape(1, -1)
  weights = np.repeat(weights, 10)
  weights = param.reshape(1, -1)
  qct = RandomTemplate(weights, num_qubits=2, seed=seed)
  qc = qct.create_circuit(qv)
  print(qc)

  return ProbsMeasure([0], QProg() << qc, qvm, qv) 


if 'grid queryer':
  @timer
  def grid_query_2d(f:Callable):
    X, Y, O = [], [], []

    #rng = np.linspace(-1, 1, args.n_plot)
    rng = np.linspace(-np.pi, np.pi, args.n_plot)
    for x in rng:
      for y in rng:
        xy = np.asarray([x, y])
        o, _ = f(xy)
        X .append(x)
        Y .append(y)
        O.append(o)
    return [np.asarray(e) for e in [X, Y, O]]

  @timer
  def grid_query_3d(f:Callable):
    X, Y, Z = [], [], []
    O0, O1  = [], []

    rng = np.linspace(-np.pi, np.pi, args.n_plot)
    for x in rng:
      for y in rng:
        for z in rng:
          xyz = np.asarray([x, y, z])
          o0, o1 = f(xyz)
          X .append(x)
          Y .append(y)
          Z .append(z)
          O0.append(o0)
          O1.append(o1)
    return [np.asarray(e) for e in [X, Y, Z, O0, O1]]


def plot_qc_func(args):
  g = lambda x: np.asarray(globals()[f'qc_{args.name}'](x))

  X, Y, O = grid_query_2d(g)
  if 'stats':
    print('max:', O.max())
    print('min:', O.min())
    print('avg:', O.mean())
    print('std:', O.std())

    plt.clf()
    plt.hist(O.flatten(), bins=100)
    plt.suptitle('vis_qc_func_hist')
    plt.savefig(TMP_PATH / 'vis_qc_func_hist.png', dpi=600)

  rng = O.max() - O.min()
  if rng < 1e-5: O = np.zeros_like(O) + (rng / 2)

  plt.clf()
  plt.scatter(X, Y, s=10, c=O, alpha=0.7, cmap='bwr')
  plt.suptitle('vis_qc_func')
  plt.savefig(TMP_PATH / 'vis_qc_func.png', dpi=600)
  plt.show()

def plot_qc_f(args):   # use the pre-defined `f`
  from vis_qc_grad import f

  X, Y, Z, O0, O1 = grid_query_3d(f)
  Omax = max(O0.max(), O1.max())
  Omin = min(O0.min(), O1.min())
  O0 = (O0 - Omin) / (Omax - Omin)
  O1 = (O1 - Omin) / (Omax - Omin)

  for i in range(2):
    O = locals()[f'O{i}']
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, Z, zdir='z', s=10, c=O, alpha=0.7, cmap='bwr')
    plt.suptitle(f'vis_qc_func_f_{i}')
    plt.savefig(TMP_PATH / f'vis_qc_func_f_{i}.png', dpi=600)
    plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-N', '--n_plot', type=int, default=100, help='grid sample and plot the qcircuit function')
  parser.add_argument('-C', '--name', default='simple', help='pre-defined qcircuit name')
  parser.add_argument('--f', action='store_true', help='use the 3-param qcircuit pre-defined in vis_qc_grad.py')
  args = parser.parse_args()

  if args.f:
    args.n_plot = 30
    plot_qc_f(args)
  else:
    plot_qc_func(args)
