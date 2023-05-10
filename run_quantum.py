#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/09 

from argparse import ArgumentParser
from traceback import print_exc
import warnings ; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import pyqpanda as pq
import pyvqnet as vq



if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-L', '--analyzer', choices=['char', '2gram', '3gram', 'kgram'], help='tokenize level')
  parser.add_argument('-M', '--model', choices=['QNN'], help='model arch')
  parser.add_argument('--name', help='exp name (optional)')
  parser.add_argument('--eval', action='store_true', help='compare result scores')
  args = parser.parse_args()
