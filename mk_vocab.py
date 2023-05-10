#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/09 

# heuristically build a k-gram vocab dictionary from `train.txt`

import math
from pathlib import Path
from functools import reduce, partial
from argparse import ArgumentParser
from collections import defaultdict, Counter, OrderedDict
from typing import Callable, Dict, Union, Tuple, List, Optional

from wordcloud import WordCloud

from utils import LOG_PATH, load_dataset, timer

''' vocab '''

Vocab  = Dict[str, int]
VocabP = Dict[str, float]

def load_vocab(fp:str) -> Vocab:
  with open(fp, encoding='utf-8') as fh:
    lines = fh.read().rstrip().split('\n')
  val_cnt = [line.split('\t') for line in lines]
  return {v: int(c) for v, c in val_cnt}

def dump_vocab(voc:Vocab, fp:str, sort:bool=False):
  if sort: voc = sort_vocab(voc)

  wc = WordCloud(font_path='simhei.ttf', height=1600, width=2048, background_color='white')
  wc.fit_words(voc)
  wc.to_file(Path(fp).with_suffix('.png'))

  with open(fp, 'w', encoding='utf-8') as fh:
    for v, c in voc.items():
      fh.write(f'{v}\t{c}\n')

def sort_vocab(voc:Vocab) -> Vocab:
  pairs = sorted([(c, v) for v, c in voc.items()], reverse=True)
  return OrderedDict([(v, c) for c, v in pairs])

def reverse_vocab(voc:Vocab) -> Vocab:
  return {v[::-1]: c for v, c in voc.items()}

def truncate_vocab(voc:Vocab, min_freq:int=3) -> Vocab:
  return {v: c for v, c in voc.items() if c >= min_freq}
  
def vocab_to_vocabp(voc:Vocab) -> VocabP:
  cnt = sum(voc.values())
  for v in voc: voc[v] /= cnt
  return voc

''' ngram '''

def make_ngram(n:int=2, line_parser:Callable=list):
  T, _ = load_dataset('train', normalize=True)
  voc = defaultdict(int)
  for t in T:
    chars = line_parser(t)
    for i in range(len(chars)-n):
      gram = ''.join(chars[i:i+n])
      voc[gram] += 1

  out_dp = LOG_PATH / f'{n}gram'
  out_dp.mkdir(exist_ok=True, parents=True)
  dump_vocab(voc, out_dp / 'vocab.txt', sort=True)

''' kgram '''

Term = Union[float, None]
Node = Tuple['Trie', Term]
Trie = Dict[str, Node]
TokenizedS = List[str]
TokenizedP = List[Tuple[float, List[str]]]
Tokenized = Union[TokenizedS, TokenizedP]

def _mk_trie(vocab:VocabP) -> Trie:
  ''' build a trie tree '''

  def _new_node() -> Node:
    return [{}, None]
  def _add_word(trie:Trie, word:str, p:float):
    node = trie[None]
    for ch in list(word):
      if ch not in node[0]:
        node[0][ch] = _new_node()
      node = node[0][ch]
    assert node[1] is None
    node[1] = p
  
  trie = { None: _new_node() }
  for w, p in vocab.items(): _add_word(trie, w, p)
  return trie

def _q_trie(trie:Trie, sent:str) -> List[Tuple[str, float]]:
  ''' query a trie tree with given sentence, return all possible prefixing word and its prob '''
  
  node = trie[None]
  no_w = sent[:1]     # in case not present in train set
  no_p = 1e-8         # FIXME: hard-coded magic number
  NO_MATCH = (no_w, no_p)
  
  words, gram = [NO_MATCH], ''
  for ch in list(sent):
    if ch not in node[0]: break
    node = node[0][ch]
    gram += ch
    if node[1] is not None:
      words.append((gram, node[1]))
  
  return words

def _tokenize(trie:Trie, sent:str, n_beam:int=3, top_k:int=None) -> Tokenized:
  '''
    uni-directional tokenizer with beam search
    NOTE: when top_k is None it returns [words], otherwise [(prob, [words])]
  '''

  candidates: List[float, str, List[str]] = [
    # Σlog(p), sent_remnant, toks
    [0.0, sent, []],
  ]

  updated = True
  while updated:
    updated = False

    candidates_new = []
    for logp, sent, toks in candidates:
      if not sent:  # just copy
        candidates_new.append([logp, sent, toks])
        continue

      init_words = _q_trie(trie, sent)
      init_words = sorted(init_words, key=lambda e: e[1], reverse=True)[:n_beam]

      for w, p in init_words:
        sent_new = sent[len(w):]
        candidates_new.append([
          logp + math.log(p),
          sent_new,
          toks + [w],
        ])
        updated = True

    candidates = sorted(candidates_new, reverse=True)[:n_beam]

  if top_k is None:
    return candidates[0][-1]    # NOTE: only keep the solution with highest prob
  else:
    return [(cand[0], cand[-1]) for cand in candidates[:top_k]]   

def _tokenize_bidirectional(trie:Trie, trie_rev:Trie, sent:str, n_beam:int=3, top_k:int=None) -> Tokenized:
  ''' wraps _tokenize() '''

  res     = _tokenize(trie,     sent,       n_beam, top_k or 1)
  res_rev = _tokenize(trie_rev, sent[::-1], n_beam, top_k or 1)
  res_rev = [(prob, [w[::-1] for w in words][::-1]) for prob, words in res_rev]
  res = sorted(res + res_rev, reverse=True)
  if 'dedup':
    res = [(round(p, ndigits=5), tuple(ws)) for p, ws in res]
    res = sorted(set(res), reverse=True)
    res = [(p, list(ws)) for p, ws in res]

  if top_k is None: return res[0][-1]
  if top_k  > 0:    return res[:top_k]
  if top_k <= 0:    return res

def make_tokenizer(fp_or_vocab:Union[str, Vocab, VocabP]=None, bidrectional:bool=True) -> Callable[[str, Optional[int], Optional[int]], Tokenized]:
  ''' use a vocab to build a tokenizer '''

  if isinstance(fp_or_vocab, Dict):
    vocab = fp_or_vocab
  else:
    if isinstance(fp_or_vocab, (str, Path)):
      fp = fp_or_vocab
    elif fp_or_vocab is None:
      fp = LOG_PATH / 'kgram' / 'vocab.txt'
    else: raise ValueError
    vocab = load_vocab(fp)
  
  assert len(vocab), 'vocab should not be empty'
  probs = list(vocab.values())
  if not isinstance(probs[0], float):
    vocab = vocab_to_vocabp(vocab)

  if bidrectional:
    trie     = _mk_trie(              vocab)
    trie_rev = _mk_trie(reverse_vocab(vocab))
    return partial(_tokenize_bidirectional, trie, trie_rev)
  else:
    trie = _mk_trie(vocab)
    return partial(_tokenize, trie)

@timer
def make_kgram(vocabs:List[Vocab], min_freq:int=3, n_beam:int=4):
  # reverse list & turn freq to prob & merge all vocabs
  for i, voc in enumerate(vocabs):
    vocabs[i] = vocab_to_vocabp(truncate_vocab(voc, min_freq))
  vocab_uni = reduce(lambda ret, voc: ret.update(voc) or ret, vocabs, {})

  # make tokenizer & load text & tokenize
  tokenizer = make_tokenizer(vocab_uni)
  T, _ = load_dataset('train', normalize=True)
  T_toks: List[str] = reduce(lambda ret, sent: ret.extend(tokenizer(sent, n_beam)) or ret, T, [])

  # collect all tokens as the new vocab
  out_dp = LOG_PATH / 'kgram'
  out_dp.mkdir(exist_ok=True, parents=True)
  dump_vocab(truncate_vocab(Counter(T_toks), min_freq), out_dp / 'vocab.txt', sort=True)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--ngram', action='store_true', help='make ngram vocab')
  parser.add_argument('--kgram', action='store_true', help='make kgram vocab')
  parser.add_argument('--min_freq', default=3, type=int, help='kgram min_freq')
  parser.add_argument('--n_beam',   default=3, type=int, help='kgram n_beam')
  args = parser.parse_args()

  Ns = [2, 3, 4, 5]

  if args.ngram:
    # fixed n-gram
    for n in Ns:
      print(f'>> making {n}gram ...')
      make_ngram(n)

  if args.kgram:
    # adaptive k-gram
    vocabs = []
    for n in Ns:
      vocabs.append(load_vocab(LOG_PATH / f'{n}gram' / 'vocab.txt'))
    print('>> making kgram ...')
    make_kgram(vocabs, min_freq=args.min_freq, n_beam=args.n_beam)
