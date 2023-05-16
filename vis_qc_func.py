#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/14 

from argparse import ArgumentParser
from typing import Callable

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from pyqpanda import CPUQVM, QCircuit, QProg
from pyqpanda import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, CR, CP
from pyvqnet.qnn import ProbsMeasure
from pyqpanda import deep_copy

from utils import TMP_PATH, timer, savefig, NDArray, Qubits

# vqc in a fucntional view


def qc_simple(qv:Qubits, param:NDArray) -> QCircuit:
  qc = QCircuit() \
    << RX(qv[0], param[0]) \
    << RX(qv[0], pi/3)
  return qc

def qc_complex(qv:Qubits, param:NDArray) -> QCircuit:
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
  return qc

def qc_test(qv:Qubits, param:NDArray) -> QCircuit:
  qc = QCircuit() \
    << RX(qv[0], param[0]) \
    << RY(qv[1], param[1]) \
    << CNOT(qv[0], qv[1]) \
    << RX(qv[1], param[2])
  return qc

def qc_toy(qv:Qubits, param:NDArray) -> QCircuit:
  ''' the concrete 4qubit-1repeat [RY]-CNOT-[RY] toy YouroQNet '''

  tht = param
  #psi = np.linspace(pi/4, pi/2, 8)
  np.random.seed(114514)
  psi = np.random.normal(size=[8])

  qc = QCircuit()
  qc << RY(qv[0], psi[0])
  qc << RY(qv[1], tht[0]) \
     << RY(qv[2], tht[1]) \
     << RY(qv[3], tht[2])
  qc << CNOT(qv[1], qv[2]) \
     << CNOT(qv[2], qv[3]) \
     << CNOT(qv[3], qv[1])
  qc << RY(qv[1], psi[1]).control(qv[0]) \
     << RY(qv[2], psi[2]).control(qv[0]) \
     << RY(qv[3], psi[3]).control(qv[0])
  qc << RY(qv[0], psi[4])
  qc << RY(qv[0], psi[5]).control(qv[1]) \
     << RY(qv[0], psi[6]).control(qv[2]) \
     << RY(qv[0], psi[7]).control(qv[3])
  return qc


if 'grid queryer':
  @timer
  def grid_query_2d(f:Callable):
    X, Y, O = [], [], []

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
    X, Y, Z, O = [], [], [], []

    rng = np.linspace(-np.pi, np.pi, args.n_plot)
    for x in rng:
      for y in rng:
        for z in rng:
          xyz = np.asarray([x, y, z])
          o, _ = f(xyz)
          X .append(x)
          Y .append(y)
          Z .append(z)
          O.append(o)
    return [np.asarray(e) for e in [X, Y, Z, O]]

  def stats_O(O:np.array, name:str):
    print('max:', O.max())
    print('min:', O.min())
    print('avg:', O.mean())
    print('std:', O.std())

    plt.clf()
    plt.hist(O.flatten(), bins=100)
    plt.suptitle('vis_qc_func_hist')
    savefig(TMP_PATH / f'vis_qc_func_{name}_hist.png')


def plot_qc_func(args):
  qct = globals()[f'qc_{args.name}']
  if 'print_circuit_once':
    print(qct(qv, np.zeros([args.dim])))
  f = lambda x: np.asarray(ProbsMeasure([0], QProg() << qct(qv, x), qvm, qv))

  if args.dim == 2:
    X, Y, O = grid_query_2d(f)
    stats_O(O, args.name)

    rng = O.max() - O.min()
    if rng < 1e-5: O = np.zeros_like(O) + (rng / 2)

    plt.clf()
    plt.scatter(X, Y, s=10, c=O, alpha=0.7, cmap='bwr')
    plt.suptitle(f'vis_qc_func_{args.name}')
    savefig(TMP_PATH / f'vis_qc_func_{args.name}.png')
    plt.show()

  elif args.dim == 3:
    args.n_plot = 30

    X, Y, Z, O = grid_query_3d(f)
    stats_O(O, args.name)

    rng = O.max() - O.min()
    if rng < 1e-5: O = np.zeros_like(O) + (rng / 2)

    plt.clf()
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, Z, zdir='z', s=10, c=O, alpha=0.7, cmap='bwr')
    plt.suptitle(f'vis_qc_func_{args.name}')
    savefig(TMP_PATH / f'vis_qc_func_{args.name}.png')
    plt.show()


if __name__ == '__main__':
  NAMES = [name[len('qc_'):] for name in globals() if name.startswith('qc_')]

  parser = ArgumentParser()
  parser.add_argument('-N', '--n_plot', type=int, default=100, help='grid sample density')
  parser.add_argument('-C', '--name', default='toy', choices=NAMES, help='pre-defined qcircuit name')
  args = parser.parse_args()

  configs = {
    # (n_qbuit, n_dim)
    'simple':  (2, 2),
    'complex': (2, 2),
    'toy':     (4, 3),
    'f':       (2, 3),
  }
  args.qubit, args.dim = configs[args.name]

  print_circuit_once = True

  if 'qvm':
    qvm = CPUQVM()
    qvm.init_qvm()
    qv = qvm.qAlloc_many(args.qubit)

  print(vars(args))
  plot_qc_func(args)
