#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/16 

from argparse import ArgumentParser
from functools import reduce

import sympy as sp
from sympy import symbols, simplify, expand
from sympy import sin, cos, pi, E as e, I as i
import numpy as np

# vqc in a apriori / mathematical / logical / computaional architecture view

if 'Tiny-Q style gates, but symbolic':
  # https://en.wikipedia.org/wiki/List_of_quantum_logic_gates#Rotation_operator_gates
  S = lambda x: symbols(x, real=True)   # raw inputs are assumed real
  RX = lambda x: np.asarray([
    [cos(S(x)/2), -i*sin(S(x)/2)],
    [-i*sin(S(x)/2), cos(S(x)/2)],
  ])
  RY = lambda x: np.asarray([
    [cos(S(x)/2), -sin(S(x)/2)],
    [sin(S(x)/2),  cos(S(x)/2)],
  ])
  RZ = lambda x: np.asarray([
    [e**(-i*S(x)/2), 0],
    [0, e**(i*S(x)/2)],
  ])
  CNOT = np.asarray([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
  ])
  CRY = lambda x: np.asarray([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, cos(S(x)/2), -sin(S(x)/2)],
    [0, 0, sin(S(x)/2),  cos(S(x)/2)],
  ])
  SWAP = np.asarray([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
  ])
  I = np.asarray([
    [1, 0],
    [0, 1],
  ])
  H = np.asarray([
    [1,  1],
    [1, -1],
  ]) / np.sqrt(2)

  Gate = np.ndarray
  compose = lambda *args: reduce(np.kron, args[1:], args[0])
  gate_simplify = lambda g: np.asarray([simplify(e) for e in g.flatten()]).reshape(g.shape)

  Is = { }
  def get_I(n:int) -> Gate:
    if n not in Is:
      In = I
      while n > 1:
        In = np.kron(In, I)
        n -= 1
      Is[n] = In
    return Is[n]

  # approx gates
  sin_hat = lambda x: 2.4 / pi * x
  cos_hat = lambda x: 1 - (2 / pi * x)**2
  RX_approx = lambda x: np.asarray([
    [cos_hat(S(x)/2), -i*sin_hat(S(x)/2)],
    [-i*sin_hat(S(x)/2), cos_hat(S(x)/2)],
  ])
  RY_approx = lambda x: np.asarray([
    [cos_hat(S(x)/2), -sin_hat(S(x)/2)],
    [sin_hat(S(x)/2),  cos_hat(S(x)/2)],
  ])
  RZ_approx = lambda x: np.asarray([
    [cos_hat(S(x)/2)-i*sin_hat(S(x)/2), 0],
    [0, cos_hat(S(x)/2)+i*sin_hat(S(x)/2)],
  ])
  CRY_approx = lambda x: np.asarray([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, cos_hat(S(x)/2), -sin_hat(S(x)/2)],
    [0, 0, sin_hat(S(x)/2),  cos_hat(S(x)/2)],
  ])


def run_circuit(qc:Gate, calc_pp=True):
  # can be very very slow, can not estimate run time!!
  #qc = gate_simplify(qc)

  # [nq**2, nq**2]
  print('qc.shape:', qc.shape)
  nc = qc.shape[0]
  nq = int(np.sqrt(nc))

  # init state: |0000>
  q = np.zeros([nc]) ; q[0] = 1
  # final state: |p>, [nc]
  p = qc @ q
  print('p.shape:', p.shape)

  # slow, but we need the output!!
  print('>> simplifying coeffs of p[k], this will take long ...')
  for i in range(nc):
    p[i] = simplify(p[i])
    print(f'|{bin(i)[2:].rjust(nq, "0")}>:', p[i])

  if calc_pp:
    print('>> simplifying probs of p[k], this will take long ...')
    pp = np.empty_like(p)
    for i in range(nc):
      pp[i] = simplify(sp.Abs(p[i])**2)
      print(f'|{bin(i)[2:].rjust(nq, "0")}>:', pp[i])

  # write to global env
  globals()['qc']  = qc
  globals()['p']   = p
  if calc_pp:
    globals()['pp']  = pp


def go_single_qubit_encoder():
  # the single-qubit data encoder:
  #  - 1 qubit
  #  - 1~3 param
  #  - rotate with RX/RY/RZ
  #  - no entangle
  # circuit:     c0         c1         c2
  #          ┌────────┐ ┌────────┐ ┌────────┐ 
  # q_0: |0>─┤RX(tht0)├─┤RY(tht1)├─┤RZ(tht2)├─
  #          └────────┘ └────────┘ └────────┘ 

  def build_circuit(RX:Gate, RY:Gate, RZ:Gate) -> Gate:
    c0 = RX('θ_0')
    c1 = RY('θ_1')
    c2 = RZ('θ_2')
    qc = c2 @ c1 @ c0
    return qc
  
  # run the circuit
  run_circuit(build_circuit(RX, RY, RZ))

  # coeffs on:
  # |0>: ( I*sin(θ_0/2)*sin(θ_1/2) + cos(θ_0/2)*cos(θ_1/2))*exp(-I*θ_2/2)
  # |1>: (-I*sin(θ_0/2)*cos(θ_1/2) + sin(θ_1/2)*cos(θ_0/2))*exp( I*θ_2/2)
  # probs on:
  # |0>:  0.5*cos(θ_0)*cos(θ_1) + 0.5
  # |1>: -0.5*cos(θ_0)*cos(θ_1) + 0.5

  # approx circuit
  p0 =  0.5*cos_hat(S('x')) * cos_hat(S('y')) + 0.5
  p1 = -0.5*cos_hat(S('x')) * cos_hat(S('y')) + 0.5
  print('approximated:')
  print('p0:', expand(simplify(p0)))
  print('p1:', expand(simplify(p1)))

  # probs on:
  # |0>: + 8*x^2*y^2/pi^4 - 2*x^2/pi^2 - 2*y^2/pi^2 + 1
  # |1>: - 8*x^2*y^2/pi^4 + 2*x^2/pi^2 + 2*y^2/pi^2


def go_YouroQNet_toy():
  # the YouroQNet toy:
  #  - 4 qubits
  #  - 11 param (tht=3, psi=8)
  #  - rotate with RY
  #  - entangle with CNOT / CRY
  # circuit:     c0       c1     c2     c3      c4         c5         c6         c7         c8         c9         c10
  #          ┌────────┐                                                       ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  # q_0: |+>─┤RY(psi0)├ ────── ────── ────── ────■───── ────■───── ────■───── ┤RY(psi4)├ ┤RY(psi5)├ ┤RY(psi6)├ ┤RY(psi7)├
  #          ├────────┤               ┌────┐ ┌───┴────┐     │          │      └────────┘ └────┬───┘ └───┬────┘ └───┬────┘
  # q_1: |0>─┤RY(tht0)├ ───■── ────── ┤CNOT├ ┤RY(psi1)├ ────┼───── ────┼───── ────────── ─────■──── ────┼───── ────┼─────
  #          ├────────┤ ┌──┴─┐        └──┬─┘ └────────┘ ┌───┴────┐     │                                │          │
  # q_2: |0>─┤RY(tht1)├ ┤CNOT├ ───■── ───┼── ────────── ┤RY(psi2)├ ────┼───── ────────── ────────── ────■───── ────┼─────
  #          ├────────┤ └────┘ ┌──┴─┐    │              └────────┘ ┌───┴────┐                                      │
  # q_3: |0>─┤RY(tht2)├ ────── ┤CNOT├ ───■── ────────── ────────── ┤RY(psi3)├ ────────── ────────── ────────── ────■─────
  #          └────────┘        └────┘                              └────────┘                

  def build_circuit(RY:Gate, CRY:Gate) -> Gate:
    # circuit axiliary
    I2 = get_I(2)
    I3 = get_I(3)
    swap01 = compose(SWAP, I2)
    swap12 = compose(I, SWAP, I)
    swap23 = compose(I2, SWAP)

    # circuit clocks
    c0  = compose(RY('ψ_0'), RY('θ_0'), RY('θ_1'), RY('θ_2'))
    c1  = compose(I, CNOT, I)
    c2  = compose(I2, CNOT)
    c3  = (swap12 @ swap23) @ compose(I2, CNOT) @ (swap23 @ swap12)
    c4  = compose(CRY('ψ_1'), I2)
    c5  = swap23 @ compose(CRY('ψ_2'), I2) @ swap23
    c6  = (swap23 @ swap12) @ compose(CRY('ψ_3'), I2) @ (swap12 @ swap23)
    c7  = compose(RY('ψ_4'), I3)
    c8  = swap01 @ compose(CRY('ψ_5'), I2) @ swap01
    c9  = (swap12 @ swap01) @ compose(CRY('ψ_6'), I2) @ (swap01 @ swap12)
    c10 = (swap23 @ swap12 @ swap01) @ compose(CRY('ψ_7'), I2) @ (swap01 @ swap12 @ swap23)

    # circuit (chaining up all gates)
    qc = c10 @ c9 @ c8 @ c7 @ c6 @ c5 @ c4 @ c3 @ c2 @ c1 @ c0 @ compose(H, I3)
    return qc

  # run the accurate circuit
  #run_circuit(build_circuit(RY, CRY), calc_pp=False)

  # coeff α on component α|0000> (eqv. matrix cell of qc[0, 0]):
  # - sin(θ_0/2)*sin(θ_1/2)*sin(θ_2/2) * sin(ψ_0/2 + pi/4) * cos(ψ_1/2 + ψ_2/2)*cos(ψ_3/2) * sin(ψ_4/2)
  # + sin(θ_0/2)*sin(θ_1/2)*cos(θ_2/2) * sin(ψ_0/2 + pi/4) * sin(ψ_1/2 + ψ_2/2)*sin(ψ_3/2) * sin(ψ_4/2)
  # + cos(θ_0/2)*cos(θ_1/2)*sin(θ_2/2) * sin(ψ_0/2 + pi/4) * sin(ψ_1/2 + ψ_2/2)*cos(ψ_3/2) * sin(ψ_4/2) 
  # - cos(θ_0/2)*cos(θ_1/2)*cos(θ_2/2) * sin(ψ_0/2 + pi/4) * cos(ψ_1/2 + ψ_2/2)*sin(ψ_3/2) * sin(ψ_4/2)
  # + sin(θ_0/2)*sin(θ_1/2)*sin(θ_2/2) * cos(ψ_0/2 + pi/4)                                 * cos(ψ_4/2)

  # Hence when param values are restricted in range [-pi, pi], the actual value range for the inner `sin()/cos()` will be [-pi/2, pi/2]
  # Within this range, function `sin()/cos()` is semi-linear :), you can draw plots here: https://www.desmos.com/calculator)
  # We try to approximate it just by linear & quadra:
  #   sin(x) -> k/π * x, where k ~= 2.4
  #   cos(x) -> 1 - (2/π * x)^2
  # Then this long-long p[0] will become a multi-variable polynomial :)

  # run the approx circuit
  run_circuit(build_circuit(RY_approx, CRY_approx))

  # (
  # -0.85*pi**6 *  ψ_0      *(ψ_4**2-C) * (θ_0**2-C)*(θ_1**2-C)*(θ_2**2-C)
  # -0.71*pi**5 * (ψ_0**2-C)*(ψ_4**2-C) * (θ_0**2-C)*(θ_1**2-C)*(θ_2**2-C)
  # -2.21*pi**5 *  ψ_0      * ψ_4       *  θ_0        * θ_1        * θ_2         * ψ_3      *(1.44*C*ψ_1*ψ_2-(ψ_1**2-C)*(ψ_2**2-C))
  # +1.76*pi**4 * (ψ_0**2-C)* ψ_4       *  θ_0        * θ_1        * θ_2         * ψ_3      *(1.44*C*ψ_1*ψ_2-(ψ_1**2-C)*(ψ_2**2-C))
  # -1.76*pi**4 *  ψ_0      * ψ_4       * (θ_0*θ_1*(θ_2**2-C)*(ψ_3**2-C)-θ_2*ψ_3*(θ_0**2-C)*(θ_1**2-C)) * (ψ_1*(ψ_2**2-C) + ψ_2*(ψ_1**2-C)) 
  # +1.47*pi**3 * (ψ_0**2-C)* ψ_4       * (θ_0*θ_1*(θ_2**2-C)*(ψ_3**2-C)-θ_2*ψ_3*(θ_0**2-C)*(θ_1**2-C)) * (ψ_1*(ψ_2**2-C) + ψ_2*(ψ_1**2-C))
  # +1.02*pi    *  ψ_0      * ψ_4       * (θ_0**2-C)*(θ_1**2-C)*(θ_2**2-C)       *(ψ_3**2-C)*(1.44*C*ψ_1*ψ_2-(ψ_1**2-C)*(ψ_2**2-C)) 
  # -0.85       * (ψ_0**2-C)* ψ_4       * (θ_0**2-C)*(θ_1**2-C)*(θ_2**2-C)       *(ψ_3**2-C)*(1.44*C*ψ_1*ψ_2-(ψ_1**2-C)*(ψ_2**2-C)) 
  # )/pi**15, where C = pi**2

  # Even further collapse all multiplier constants to get the functional form ——
  # => Fold all `psi` to const, get the partial function form of f(tht):
  #   f(tht) = (θ_0*θ_1 + (θ_0**2-C)*(θ_1**2-C)) * (θ_2 + (θ_2**2-C))
  # where C are different consts
  # now it's clear to see the charateristic function is a 3-variable power-6 polynomial function :)
  # accordingly, the psi params controlls the coefficient of each term in a sophisticated way...
  # => Fold all `tht` to const, get the partial function form of g(psi):
  #   g(psi) = ((ψ_0**2-C)+ψ_0)*((ψ_4**2-C)+ψ_4*(((ψ_3**2-C)-ψ_3) * (ψ_1*(ψ_2**2-C) + ψ_2*(ψ_1**2-C))+(ψ_3+(ψ_3**2-C))*(ψ_1*ψ_2-(ψ_1**2-C)*(ψ_2**2-C))))


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='toy', choices=['enc', 'toy'])
  args = parser.parse_args()

  if args.model == 'enc':
    go_single_qubit_encoder()
  if args.model == 'toy':
    go_YouroQNet_toy()

  print()
  print('-' * 72)
  import code
  code.interact(local=globals())
