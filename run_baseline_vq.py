#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/05 

import pickle as pkl
from pathlib import Path
from argparse import ArgumentParser
from traceback import print_exc
import warnings ; warnings.simplefilter("ignore")

if 'pyvqnet':
  import pyvqnet as vq
  # basics
  from pyvqnet.nn.parameter import Parameter
  from pyvqnet.nn.module import Module
  from pyvqnet.nn.pixel_shuffle import Pixel_Shuffle, Pixel_Unshuffle
  # linear
  from pyvqnet.nn.embedding import Embedding
  from pyvqnet.nn.linear import Linear
  # conv
  from pyvqnet.nn.conv import Conv2D, Conv1D, ConvT2D
  from pyvqnet.nn.pooling import MaxPool1D, MaxPool2D, AvgPool1D, AvgPool2D
  # recurrent
  from pyvqnet.nn.rnn import RNN, Dynamic_RNN
  from pyvqnet.nn.gru import GRU, Dynamic_GRU
  from pyvqnet.nn.lstm import LSTM, Dynamic_LSTM
  # net structure
  from pyvqnet.nn.self_attention import Self_Conv_Attention
  from pyvqnet.nn.transformer import Transformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, MultiHeadAttention
  # regularity
  from pyvqnet.nn.dropout import Dropout
  from pyvqnet.nn.batch_norm import BatchNorm1d, BatchNorm2d
  from pyvqnet.nn.layer_norm import LayerNorm1d, LayerNorm2d, LayerNormNd
  from pyvqnet.nn.spectral_norm import Spectral_Norm
  # activation
  from pyvqnet.nn.activation import Sigmoid, ReLu, LeakyReLu, Softmax, Softplus, Softsign, HardSigmoid, ELU, Tanh
  # optimizing
  from pyvqnet.nn.loss import CategoricalCrossEntropy, BinaryCrossEntropy, SoftmaxCrossEntropy, CrossEntropyLoss
  from pyvqnet.optim import SGD, Adam

from utils import *


class TextDNN(Module):

  def __init__(self):
    pass

  def forward(self):
    pass

class TextCNN(Module):

  def __init__(self):
    pass

  def forward(self):
    pass

class TextRNN(Module):

  def __init__(self):
    pass

  def forward(self):
    pass


def get_model(name):
  pass


def test():
  pass


def train(name, model, datasets, logger):
  (X_train, Y_train), (X_test,  Y_test), (X_valid, Y_valid) = datasets

  model.fit(X_train, Y_train)

  precs, recalls, f1s, cmats = [], [], [], []

  logger.info(f'[{name}]')
  for split in SPLITS:
    logger.info(f'<{split}>')

    X_split = locals().get(f'X_{split}')
    Y_split = locals().get(f'Y_{split}')
    Y_pred = model.predict(X_split)

    cmat = confusion_matrix(Y_split, Y_pred)

    logger.info('>> confusion_matrix:')
    logger.info(cmat)
  
    logger.info('-' * 32)
  logger.info('\n')

  return precs, recalls, f1s, cmats


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--analyzer', choices=['char', '2gram', '3gram'], help='tokenize level')
  parser.add_argument('--model', choices=['DNN', 'CNN', 'RNN'], help='model arch')
  parser.add_argument('--name', help='exp name (optional)')
  parser.add_argument('--eval', action='store_true', help='compare result scores')
  args = parser.parse_args()

  exit(0)

  out_dp: Path = LOG_PATH / args.analyzer / (args.name or args.model)
  out_dp.mkdir(exist_ok=True, parents=True)

  train_set = load_dataset('train', normalize=True)
  test_set  = load_dataset('test' , normalize=True)
  valid_set = load_dataset('valid', normalize=True)

  logger = get_logger(out_dp / 'run.log', mode='w')
  result = { }

  raise NotImplementedError

  with open(out_dp / 'result.pkl', 'wb') as fh:
    pkl.dump(result, fh)
