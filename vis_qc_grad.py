#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/21 

from argparse import ArgumentParser
from typing import Callable

import numpy as np

from pyqpanda import CPUQVM, QCircuit, QProg
from pyqpanda import H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, CR, CP
from pyvqnet.qnn import ProbsMeasure, grad as grad_param_shift

from utils import timer

# error comparation:
#   |  dx  |   grad error   |
#   | 1e-1 | 8.269716928-03 |
#   | 1e-2 | 8.273408020-04 |
#   | 1e-3 | 8.273421272-05 |
#   | 1e-4 | 8.273419231-06 |
#   | 1e-5 | 8.273417650-07 |
#   | 1e-6 | 8.273372760-08 |


if 'QCircuit as function f()':
  qvm = CPUQVM()
  qvm.init_qvm()
  qv = qvm.qAlloc_many(2)

  def pqctest(param):
    global qvm, qv

    qc = QCircuit() \
      << RX(qv[0], param[0]) \
      << RY(qv[1], param[1]) \
      << CNOT(qv[0], qv[1]) \
      << RX(qv[1], param[2])
    prog = QProg() << qc

    return ProbsMeasure([1], prog, qvm, qv)

  f = lambda x: np.asarray(pqctest(x))


def grad_finite_diff(f:Callable, x:np.ndarray, dx:float=0.01) -> np.ndarray:
  '''
    approximated grad by finite_diff
    => grad.shape: [n_param, n_output]
  '''
  p_grad = []   # partial derivative on each input dim
  for i in range(len(x)):
    x_dx = x.copy() ; x_dx[i] += dx
    grad = (f(x_dx) - f(x)) / dx
    p_grad.append(grad)
  return np.stack(p_grad, axis=0)


@timer
def bench_mark(args):
  err = 0.0
  np.random.seed(args.seed)    # fix seed
  for _ in range(args.n_bench):
    x = np.random.normal(size=[3])

    g_ps = grad_param_shift(f, x)
    g_fd = grad_finite_diff(f, x, args.dx)
    err += np.abs(g_ps - g_fd).mean()

  print('[bench_mark] mean error:', err / args.n_bench)


def show_grad(args):
  x = np.random.normal(size=[3])

  # grad by param_shift
  g_ps = grad_param_shift(f, x)
  print('g_ps:')
  print(g_ps)

  # grad by finite_diff 
  g_fd = grad_finite_diff(f, x, args.dx)
  print('g_fd:')
  print(g_fd)

  print()

  # grad diff
  d_g = g_ps - g_fd
  print('d_g:')
  print(d_g)
  print('avg(|d_g|):', np.abs(d_g).mean())


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--dx',    type=float, default=0.01,   help='delta x for finite_diff')
  parser.add_argument('--n_bench', type=int, default=0,      help='bench mark the error')
  parser.add_argument('--seed',    type=int, default=114514, help='seed for bench mark')
  args = parser.parse_args()

  if args.n_bench > 0:
    bench_mark(args)
  else:
    show_grad(args)
