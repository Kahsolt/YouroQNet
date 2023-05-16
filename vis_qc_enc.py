#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/16 

# There seems a convention to encode a datapoint x using circuit:
#   |0> - RY(arctan(x)) - RZ(arctan(x^2))
# why not use the following?
#   |0> - RY(arctan(x)) - RZ(arctan(x))
#   |0> - RY(arctan(x))

# NOTE: this script uses Tiny-Q, need manually install follow https://github.com/Kahsolt/Tiny-Q

from tiny_q import *

import numpy as np
from numpy import arctan
import matplotlib.pyplot as plt

Xs = np.linspace(-10, 10, 1000)


plt.subplot(221)
plt.plot(Xs, arctan(Xs),    label='arctan(x)')
plt.plot(Xs, arctan(Xs**2), label='arctan(x**2)')
plt.title('f(x) = arctan(x)')
plt.legend()


def plot_qc_f(subloc:int, f:Callable, title:str):
  real = []
  imag = []
  for x in Xs:
    y = f(x)     # the conplex α on |0>
    real.append(y.real)
    imag.append(y.imag)

  plt.subplot(subloc)
  plt.plot(Xs, real, label='α real')
  plt.plot(Xs, imag, label='α imag')
  plt.title(title)
  plt.legend()


plot_qc_f(222, lambda x: (RY(arctan(x)) << RZ(arctan(x**2)) | v0).v[0], '|0> - RY(arctan(x)) - RZ(arctan(x**2))')
plot_qc_f(223, lambda x: (RY(arctan(x)) << RZ(arctan(x))    | v0).v[0], '|0> - RY(arctan(x)) - RZ(arctan(x))')
plot_qc_f(224, lambda x: (RY(arctan(x))                     | v0).v[0], '|0> - RY(arctan(x))')

plt.show()
