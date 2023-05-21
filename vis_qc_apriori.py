#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/16 

from argparse import ArgumentParser
from functools import reduce

from sympy import simplify
from sympy import symbols as S
from sympy import sin, cos, pi, E as e, I as i
import numpy as np

# vqc in a apriori / mathematical / logical / computaional architecture view

if 'Tiny-Q style gates, but symbolic':
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
  RY_approx = lambda x: np.asarray([
    [cos_hat(S(x)/2), -sin_hat(S(x)/2)],
    [sin_hat(S(x)/2),  cos_hat(S(x)/2)],
  ])
  CRY_approx = lambda x: np.asarray([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, cos_hat(S(x)/2), -sin_hat(S(x)/2)],
    [0, 0, sin_hat(S(x)/2),  cos_hat(S(x)/2)],
  ])


def run_circuit(qc:Gate):
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
  print('>> simplifying p[k], this will take long ...')
  for i in range(nc):
    print(f'|{bin(i)[2:].rjust(nq, "0")}>:')
    print(simplify(p[i]))

  # write to global env
  globals()['qc'] = qc
  globals()['p']  = p


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

  def build_circuit():
    c0 = RX('θ_0')
    c1 = RY('θ_1')
    c2 = RZ('θ_2')
    qc = c2 @ c1 @ c0
    return qc

  run_circuit(build_circuit())

  # prob on:
  # |0>: exp( im(θ_2)/2) * Abs(I*sin(θ_0/2)*sin(θ_1/2) + cos(θ_0/2)*cos(θ_1/2))
  # |1>: exp(-im(θ_2)/2) * Abs(I*sin(θ_0/2)*cos(θ_1/2) - sin(θ_1/2)*cos(θ_0/2))


def go_YouroQNet_toy():
  # the YouroQNet toy:
  #  - 4 qubits
  #  - 11 param (tht=3, psi=8)
  #  - rotate with RY
  #  - entangle with CNOT / CRY
  # circuit:     c0       c1     c2     c3      c4         c5         c6         c7         c8         c9         c10
  #          ┌────────┐                                                       ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  # q_0: |0>─┤RY(psi0)├ ────── ────── ────── ────■───── ────■───── ────■───── ┤RY(psi4)├ ┤RY(psi5)├ ┤RY(psi6)├ ┤RY(psi7)├
  #          ├────────┤               ┌────┐ ┌───┴────┐     │          │      └────────┘ └────┬───┘ └───┬────┘ └───┬────┘
  # q_1: |0>─┤RY(tht0)├ ───■── ────── ┤CNOT├ ┤RY(psi1)├ ────┼───── ────┼───── ────────── ─────■──── ────┼───── ────┼─────
  #          ├────────┤ ┌──┴─┐        └──┬─┘ └────────┘ ┌───┴────┐     │                                │          │
  # q_2: |0>─┤RY(tht1)├ ┤CNOT├ ───■── ───┼── ────────── ┤RY(psi2)├ ────┼───── ────────── ────────── ────■───── ────┼─────
  #          ├────────┤ └────┘ ┌──┴─┐    │              └────────┘ ┌───┴────┐                                      │
  # q_3: |0>─┤RY(tht2)├ ────── ┤CNOT├ ───■── ────────── ────────── ┤RY(psi3)├ ────────── ────────── ────────── ────■─────
  #          └────────┘        └────┘                              └────────┘                

  def build_circuit(RY:Gate, CRY:Gate):
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
    qc = c10 @ c9 @ c8 @ c7 @ c6 @ c5 @ c4 @ c3 @ c2 @ c1 @ c0
    return qc

  # run the accurate circuit
  run_circuit(build_circuit(RY, CRY))

  # coeff α on component α|0000> (eqv. matrix cell of qc[0, 0]):
  # + sin(θ_0/2)*sin(θ_1/2)*sin(θ_2/2)*sin(ψ_0/2)*sin(ψ_3/2)*sin(ψ_4/2)*cos(ψ_1/2+ψ_2/2)
  # + sin(θ_0/2)*sin(θ_1/2)*sin(ψ_0/2)*sin(ψ_4/2)*sin(ψ_1/2+ψ_2/2)*cos(θ_2/2)*cos(ψ_3/2)
  # - sin(θ_2/2)*sin(ψ_0/2)*sin(ψ_3/2)*sin(ψ_4/2)*sin(ψ_1/2+ψ_2/2)*cos(θ_0/2)*cos(θ_1/2)
  # - sin(ψ_0/2)*sin(ψ_4/2)*cos(θ_0/2)*cos(θ_1/2)*cos(θ_2/2)*cos(ψ_3/2)*cos(ψ_1/2+ψ_2/2)
  # + cos(θ_0/2)*cos(θ_1/2)*cos(θ_2/2)*cos(ψ_0/2)*cos(ψ_4/2)

  # Hence when param values are restricted in range [-pi, pi], the actual value range for the inner `sin()/cos()` will be [-pi/2, pi/2]
  # Let's eliminate all `/2`, reorder & group it (=> see also full results in `img/vis_qc_logic.txt`)

  #        theta (embed)                         psi (ansatz)
  # + sin(θ_0)*sin(θ_1)*sin(θ_2) * sin(ψ_0) * cos(ψ_1+ψ_2)*sin(ψ_3) * sin(ψ_4)
  # + sin(θ_0)*sin(θ_1)*cos(θ_2) * sin(ψ_0) * sin(ψ_1+ψ_2)*cos(ψ_3) * sin(ψ_4)
  # - cos(θ_0)*cos(θ_1)*sin(θ_2) * sin(ψ_0) * sin(ψ_1+ψ_2)*sin(ψ_3) * sin(ψ_4)
  # - cos(θ_0)*cos(θ_1)*cos(θ_2) * sin(ψ_0) * cos(ψ_1+ψ_2)*cos(ψ_3) * sin(ψ_4)
  # + cos(θ_0)*cos(θ_1)*cos(θ_2) * cos(ψ_0) *                         cos(ψ_4)

  # run the approx circuit
  run_circuit(build_circuit(RY_approx, CRY_approx))

  # Within this range [-pi/2, pi/2], function `sin()/cos()` is semi-linear :), you can draw plots here: https://www.desmos.com/calculator)
  # We try to approximate it just by linear & quadra:
  #   sin(x) -> k/π * x, where k ~= 2.4
  #   cos(x) -> 1 - (2/π * x)^2
  # Then this long-long p[0] will become a multi-variable polynomial :)

  # ( - 2.1*pi^4 *  θ_0        * θ_1        * θ_2                                         *  ψ_0        * ψ_4         * (2.1*pi^2*ψ_1*ψ_2-1.4*(ψ_1^2-pi^2)*(ψ_2^2-pi^2)) *  ψ_3
  #   -     pi^4 * (θ_0^2-pi^2)*(θ_1^2-pi^2)*(θ_2^2-pi^2)                                 * (ψ_0^2-pi^2)*(ψ_4^2-pi^2)
  #   + 1.2      * (θ_0^2-pi^2)*(θ_1^2-pi^2)*(θ_2^2-pi^2)                                 *  ψ_0        * ψ_4         * (1.7*pi^2*ψ_1*ψ_2-1.2*(ψ_1^2-pi^2)*(ψ_2^2-pi^2)) * (ψ_3^2-pi^2)
  #   + 2.5*pi^3 * (-θ_0*θ_1*(θ_2^2-pi^2)*(ψ_3^2-pi^2)+θ_2*ψ_3*(θ_0^2-pi^2)*(θ_1^2-pi^2)) *  ψ_0        * ψ_4         * (ψ_1*(ψ_2^2-pi^2)+(ψ_1^2-pi^2)*ψ_2)
  # ) / pi^14

  # Even further collapse all multiplier constants to get the functional form:

  '''
  [c**6]
  - ψ_0*ψ_4                 # the gloabl ansatz bias (AB)

  [c**5]
  + θ_0**2*ψ_0*ψ_4          # single input theta tweeks AB
  + θ_1**2*ψ_0*ψ_4 
  + θ_2**2*ψ_0*ψ_4 
  + ψ_0*ψ_1**2*ψ_4          # single ansatz psi tweeks AB
  + ψ_0*ψ_2**2*ψ_4 
  + ψ_0*ψ_3**2*ψ_4 
  + 1                       # the gloabl systematic bias (SB)

  [c**4]
  - θ_0**2*θ_1**2*ψ_0*ψ_4   # interactions of input thetas tweeks AB
  - θ_0**2*θ_2**2*ψ_0*ψ_4 
  - θ_0**2*ψ_0*ψ_1**2*ψ_4   # interactions of ansatz psis tweeks AB
  - θ_0**2*ψ_0*ψ_2**2*ψ_4 
  - θ_0**2*ψ_0*ψ_3**2*ψ_4 
  - θ_0**2                  # single input theta tweeks SB
  - θ_1**2*θ_2**2*ψ_0*ψ_4 
  - θ_1**2*ψ_0*ψ_1**2*ψ_4 
  - θ_1**2*ψ_0*ψ_2**2*ψ_4 
  - θ_1**2*ψ_0*ψ_3**2*ψ_4 
  - θ_1**2 
  - θ_2**2*ψ_0*ψ_1**2*ψ_4 
  - θ_2**2*ψ_0*ψ_2**2*ψ_4 
  - θ_2**2*ψ_0*ψ_3**2*ψ_4 
  - θ_2**2 
  - ψ_0**2                  # single ansatz psi tweeks SB
  - ψ_0*ψ_1**2*ψ_2**2*ψ_4 
  - ψ_0*ψ_1**2*ψ_3**2*ψ_4 
  + ψ_0*ψ_1*ψ_2*ψ_4 
  - ψ_0*ψ_2**2*ψ_3**2*ψ_4 
  - ψ_4**2 

  [c**3]
  + θ_0**2*θ_1**2*θ_2**2*ψ_0*ψ_4
  + θ_0**2*θ_1**2*ψ_0*ψ_1**2*ψ_4 
  + θ_0**2*θ_1**2*ψ_0*ψ_2**2*ψ_4 
  + θ_0**2*θ_1**2*ψ_0*ψ_3**2*ψ_4 
  + θ_0**2*θ_1**2                   # interactions of input thetas tweeks SB
  + θ_0**2*θ_2**2*ψ_0*ψ_1**2*ψ_4 
  + θ_0**2*θ_2**2*ψ_0*ψ_2**2*ψ_4 
  + θ_0**2*θ_2**2*ψ_0*ψ_3**2*ψ_4 
  + θ_0**2*θ_2**2 
  + θ_0**2*ψ_0**2 
  + θ_0**2*ψ_0*ψ_1**2*ψ_2**2*ψ_4 
  + θ_0**2*ψ_0*ψ_1**2*ψ_3**2*ψ_4 
  - θ_0**2*ψ_0*ψ_1*ψ_2*ψ_4 
  + θ_0**2*ψ_0*ψ_2**2*ψ_3**2*ψ_4 
  + θ_0**2*ψ_4**2 
  + θ_0*θ_1*ψ_0*ψ_1*ψ_4 
  + θ_0*θ_1*ψ_0*ψ_2*ψ_4 
  + θ_1**2*θ_2**2*ψ_0*ψ_1**2*ψ_4 
  + θ_1**2*θ_2**2*ψ_0*ψ_2**2*ψ_4 
  + θ_1**2*θ_2**2*ψ_0*ψ_3**2*ψ_4 
  + θ_1**2*θ_2**2 
  + θ_1**2*ψ_0**2 
  + θ_1**2*ψ_0*ψ_1**2*ψ_2**2*ψ_4 
  + θ_1**2*ψ_0*ψ_1**2*ψ_3**2*ψ_4 
  - θ_1**2*ψ_0*ψ_1*ψ_2*ψ_4 
  + θ_1**2*ψ_0*ψ_2**2*ψ_3**2*ψ_4 
  + θ_1**2*ψ_4**2 
  + θ_2**2*ψ_0**2 
  + θ_2**2*ψ_0*ψ_1**2*ψ_2**2*ψ_4 
  + θ_2**2*ψ_0*ψ_1**2*ψ_3**2*ψ_4 
  - θ_2**2*ψ_0*ψ_1*ψ_2*ψ_4 
  + θ_2**2*ψ_0*ψ_2**2*ψ_3**2*ψ_4 
  + θ_2**2*ψ_4**2 
  - θ_2*ψ_0*ψ_1*ψ_3*ψ_4 
  - θ_2*ψ_0*ψ_2*ψ_3*ψ_4 
  + ψ_0**2*ψ_4**2                 # interactions of ansatz psis tweeks SB
  + ψ_0*ψ_1**2*ψ_2**2*ψ_3**2*ψ_4 
  - ψ_0*ψ_1*ψ_2*ψ_3**2*ψ_4 

  [c**2]
  - θ_0**2*θ_1**2*θ_2**2*ψ_0*ψ_1**2*ψ_4 
  - θ_0**2*θ_1**2*θ_2**2*ψ_0*ψ_2**2*ψ_4 
  - θ_0**2*θ_1**2*θ_2**2*ψ_0*ψ_3**2*ψ_4 
  - θ_0**2*θ_1**2*θ_2**2 
  - θ_0**2*θ_1**2*ψ_0**2 
  - θ_0**2*θ_1**2*ψ_0*ψ_1**2*ψ_2**2*ψ_4 
  - θ_0**2*θ_1**2*ψ_0*ψ_1**2*ψ_3**2*ψ_4 
  + θ_0**2*θ_1**2*ψ_0*ψ_1*ψ_2*ψ_4 
  - θ_0**2*θ_1**2*ψ_0*ψ_2**2*ψ_3**2*ψ_4 
  - θ_0**2*θ_1**2*ψ_4**2 
  - θ_0**2*θ_2**2*ψ_0**2 
  - θ_0**2*θ_2**2*ψ_0*ψ_1**2*ψ_2**2*ψ_4 
  - θ_0**2*θ_2**2*ψ_0*ψ_1**2*ψ_3**2*ψ_4 
  + θ_0**2*θ_2**2*ψ_0*ψ_1*ψ_2*ψ_4 
  - θ_0**2*θ_2**2*ψ_0*ψ_2**2*ψ_3**2*ψ_4 
  - θ_0**2*θ_2**2*ψ_4**2 
  + θ_0**2*θ_2*ψ_0*ψ_1*ψ_3*ψ_4 
  + θ_0**2*θ_2*ψ_0*ψ_2*ψ_3*ψ_4 
  - θ_0**2*ψ_0**2*ψ_4**2 
  - θ_0**2*ψ_0*ψ_1**2*ψ_2**2*ψ_3**2*ψ_4 
  + θ_0**2*ψ_0*ψ_1*ψ_2*ψ_3**2*ψ_4 
  - θ_0*θ_1*θ_2**2*ψ_0*ψ_1*ψ_4 
  - θ_0*θ_1*θ_2**2*ψ_0*ψ_2*ψ_4 
  + θ_0*θ_1*θ_2*ψ_0*ψ_3*ψ_4 
  - θ_0*θ_1*ψ_0*ψ_1**2*ψ_2*ψ_4 
  - θ_0*θ_1*ψ_0*ψ_1*ψ_2**2*ψ_4 
  - θ_0*θ_1*ψ_0*ψ_1*ψ_3**2*ψ_4 
  - θ_0*θ_1*ψ_0*ψ_2*ψ_3**2*ψ_4 
  - θ_1**2*θ_2**2*ψ_0**2 
  - θ_1**2*θ_2**2*ψ_0*ψ_1**2*ψ_2**2*ψ_4 
  - θ_1**2*θ_2**2*ψ_0*ψ_1**2*ψ_3**2*ψ_4 
  + θ_1**2*θ_2**2*ψ_0*ψ_1*ψ_2*ψ_4 
  - θ_1**2*θ_2**2*ψ_0*ψ_2**2*ψ_3**2*ψ_4 
  - θ_1**2*θ_2**2*ψ_4**2 
  + θ_1**2*θ_2*ψ_0*ψ_1*ψ_3*ψ_4 
  + θ_1**2*θ_2*ψ_0*ψ_2*ψ_3*ψ_4 
  - θ_1**2*ψ_0**2*ψ_4**2 
  - θ_1**2*ψ_0*ψ_1**2*ψ_2**2*ψ_3**2*ψ_4 
  + θ_1**2*ψ_0*ψ_1*ψ_2*ψ_3**2*ψ_4 
  - θ_2**2*ψ_0**2*ψ_4**2 
  - θ_2**2*ψ_0*ψ_1**2*ψ_2**2*ψ_3**2*ψ_4 
  + θ_2**2*ψ_0*ψ_1*ψ_2*ψ_3**2*ψ_4 
  + θ_2*ψ_0*ψ_1**2*ψ_2*ψ_3*ψ_4 
  + θ_2*ψ_0*ψ_1*ψ_2**2*ψ_3*ψ_4 

  [c]
  + θ_0**2*θ_1**2*θ_2**2*ψ_0**2 
  + θ_0**2*θ_1**2*θ_2**2*ψ_0*ψ_1**2*ψ_2**2*ψ_4 
  + θ_0**2*θ_1**2*θ_2**2*ψ_0*ψ_1**2*ψ_3**2*ψ_4 
  - θ_0**2*θ_1**2*θ_2**2*ψ_0*ψ_1*ψ_2*ψ_4 
  + θ_0**2*θ_1**2*θ_2**2*ψ_0*ψ_2**2*ψ_3**2*ψ_4 
  + θ_0**2*θ_1**2*θ_2**2*ψ_4**2 
  - θ_0**2*θ_1**2*θ_2*ψ_0*ψ_1*ψ_3*ψ_4 
  - θ_0**2*θ_1**2*θ_2*ψ_0*ψ_2*ψ_3*ψ_4 
  + θ_0**2*θ_1**2*ψ_0**2*ψ_4**2 
  + θ_0**2*θ_1**2*ψ_0*ψ_1**2*ψ_2**2*ψ_3**2*ψ_4 
  - θ_0**2*θ_1**2*ψ_0*ψ_1*ψ_2*ψ_3**2*ψ_4 
  + θ_0**2*θ_2**2*ψ_0**2*ψ_4**2 
  + θ_0**2*θ_2**2*ψ_0*ψ_1**2*ψ_2**2*ψ_3**2*ψ_4 
  - θ_0**2*θ_2**2*ψ_0*ψ_1*ψ_2*ψ_3**2*ψ_4 
  - θ_0**2*θ_2*ψ_0*ψ_1**2*ψ_2*ψ_3*ψ_4 
  - θ_0**2*θ_2*ψ_0*ψ_1*ψ_2**2*ψ_3*ψ_4 
  + θ_0*θ_1*θ_2**2*ψ_0*ψ_1**2*ψ_2*ψ_4 
  + θ_0*θ_1*θ_2**2*ψ_0*ψ_1*ψ_2**2*ψ_4 
  + θ_0*θ_1*θ_2**2*ψ_0*ψ_1*ψ_3**2*ψ_4 
  + θ_0*θ_1*θ_2**2*ψ_0*ψ_2*ψ_3**2*ψ_4 
  - θ_0*θ_1*θ_2*ψ_0*ψ_1**2*ψ_3*ψ_4 
  - θ_0*θ_1*θ_2*ψ_0*ψ_2**2*ψ_3*ψ_4 
  + θ_0*θ_1*ψ_0*ψ_1**2*ψ_2*ψ_3**2*ψ_4 
  + θ_0*θ_1*ψ_0*ψ_1*ψ_2**2*ψ_3**2*ψ_4 
  + θ_1**2*θ_2**2*ψ_0**2*ψ_4**2 
  + θ_1**2*θ_2**2*ψ_0*ψ_1**2*ψ_2**2*ψ_3**2*ψ_4 
  - θ_1**2*θ_2**2*ψ_0*ψ_1*ψ_2*ψ_3**2*ψ_4 
  - θ_1**2*θ_2*ψ_0*ψ_1**2*ψ_2*ψ_3*ψ_4 
  - θ_1**2*θ_2*ψ_0*ψ_1*ψ_2**2*ψ_3*ψ_4 

  [1]
  - θ_0**2*θ_1**2*θ_2**2*ψ_0**2*ψ_4**2 
  - θ_0**2*θ_1**2*θ_2**2*ψ_0*ψ_1**2*ψ_2**2*ψ_3**2*ψ_4 
  + θ_0**2*θ_1**2*θ_2**2*ψ_0*ψ_1*ψ_2*ψ_3**2*ψ_4 
  + θ_0**2*θ_1**2*θ_2*ψ_0*ψ_1**2*ψ_2*ψ_3*ψ_4 
  + θ_0**2*θ_1**2*θ_2*ψ_0*ψ_1*ψ_2**2*ψ_3*ψ_4 
  - θ_0*θ_1*θ_2**2*ψ_0*ψ_1**2*ψ_2*ψ_3**2*ψ_4 
  - θ_0*θ_1*θ_2**2*ψ_0*ψ_1*ψ_2**2*ψ_3**2*ψ_4 
  + θ_0*θ_1*θ_2*ψ_0*ψ_1**2*ψ_2**2*ψ_3*ψ_4 
  - θ_0*θ_1*θ_2*ψ_0*ψ_1*ψ_2*ψ_3*ψ_4
  '''

  # Fold all `psi` to const, get the partial function form f(tht), not accurate but nearly:
  #   f(tht) = ΣiΣjΣkΣpΣqΣr tht_i^p * tht_j^q * tht_k^r, when i,j,k,p,q,r ∈ {0,1,2}
  # now it's clear to see the charateristic function is a 3-variable power-2 polynomial function :)
  # accordingly, the psi params controlls the coefficient of each term

  # Assume all `theta` to be small enough to eliminate all higher orders, in practice this will lead to network incapable :(
  #   g(tht) = C1*θ_0*θ_1*θ_2 + C2*θ_0*θ_1 + C3*θ_2 + C4


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
