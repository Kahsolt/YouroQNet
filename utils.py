#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/05 

import logging
from typing import List


def get_logger(fp:str, mode:str='a') -> logging.Logger:
  FORMATTER = logging.Formatter('%(message)s')
  logger = logging.getLogger('run')
  logger.setLevel(logging.DEBUG)
  h = logging.FileHandler(fp, mode, encoding='utf-8')
  h.setLevel(logging.DEBUG)
  h.setFormatter(FORMATTER)
  logger.addHandler(h)
  h = logging.StreamHandler()
  h.setLevel(logging.DEBUG)
  h.setFormatter(FORMATTER)
  logger.addHandler(h)
  return logger


def clean_text(texts:List[str]) -> List[str]:
  r = []
  for line in texts:
    r.append(' '.join([e.strip() for e in list(line) if e.strip()]))
  return r
