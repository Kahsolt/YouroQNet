#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/14 

from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from pyqpanda import CPUQVM, QCircuit, QProg
from pyqpanda import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, CR, CP
from pyvqnet.qnn import ProbsMeasure

from utils import TMP_PATH, timer
from vis_grad import f


def plot_simple(args):

  if 'QCircuit as function g()':
    qvm = CPUQVM()
    qvm.init_qvm()
    qv = qvm.qAlloc_many(2)

    def pqctest(param):
      nonlocal qvm, qv

      qc = QCircuit() \
        << RY(qv[0], param[0]) \
        << RZ(qv[1], param[1])
      prog = QProg() << qc

      return ProbsMeasure([1], prog, qvm, qv)

    g = lambda x: np.asarray(pqctest(x))

  @timer
  def grid_query():
    X, Y = [], [], []
    O = []

    rng = np.linspace(-np.pi, np.pi, args.n_plot)
    for x in rng:
      for y in rng:
        xy = np.asarray([x, y])
        o, _ = g(xy)
        X .append(x)
        Y .append(y)
        O.append(o)

    return [np.asarray(e) for e in [X, Y, O]]

  X, Y, O = grid_query()
  O = (O - O.min()) / (O.max() - O.min())

  plt.clf()
  plt.scatter(X, Y, s=10, c=O, alpha=0.7, cmap='bwr')
  plt.suptitle('vis_qc_func_g')
  plt.savefig(TMP_PATH / 'vis_qc_func_g.png', dpi=600)
  plt.show()


def plot_f(args):
  @timer
  def grid_query():
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

  X, Y, Z, O0, O1 = grid_query()
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
  parser.add_argument('--n_plot', type=int, default=30, help='grid sample and plot the qcircuit function')
  parser.add_argument('--f',      action='storee_true', help='use the 3-param qcircuit pre-defined in vis_qc_grad.py')
  args = parser.parse_args()

  if args.f:
    plot_f(args)
  else:
    plot_simple(args)
