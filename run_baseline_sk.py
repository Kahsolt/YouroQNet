#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/05 

import pickle as pkl
from pathlib import Path
from argparse import ArgumentParser
from traceback import print_exc
import warnings ; warnings.simplefilter("ignore")

import jieba
import numpy as np
import matplotlib.pyplot as plt

try:
  from sklearn.decomposition import PCA, KernelPCA
  from sklearn.manifold import TSNE
  try:
    # not all components are safely patchable, leave them unhacked
    from sklearnex import patch_sklearn ; patch_sklearn()
  except:
    print_exc()
    print('sklearnex not installed, performance may be slow')
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
  from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
  from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
  from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, PassiveAggressiveClassifier, SGDClassifier
  from sklearn.svm import SVC, NuSVC, LinearSVC
  from sklearn.neural_network import MLPClassifier, BernoulliRBM
  from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
except:
  print_exc()
  print('sklearn not installed, some of the features may not work')

try:
  import fasttext.util
  from fasttext.FastText import _FastText as FastText
except:
  print_exc()
  print('fasttext not installed, some of the features may not work')

from utils import *
from mk_vocab import make_tokenizer

FASTTEXT_CKPT_PATH = DATA_PATH / 'cc.zh.300.bin'

FEATURES = [
  'tfidf',
  'fasttext',
]
ANALYZERS = [
  'char',
  'word',
  'sent',
  '2gram',
  '3gram',
  'kgram',
]
MODELS = {
  'knn-5':   lambda: KNeighborsClassifier(n_neighbors=5),
  'knn-7':   lambda: KNeighborsClassifier(n_neighbors=7),
  'knn-10':  lambda: KNeighborsClassifier(n_neighbors=10),
  'knn-20':  lambda: KNeighborsClassifier(n_neighbors=20),
  'dt':      lambda: DecisionTreeClassifier(max_depth=8),
  'et':      lambda: ExtraTreeClassifier(max_depth=8),
  'rf':      lambda: RandomForestClassifier(n_estimators=16),
  'ets':     lambda: ExtraTreesClassifier(n_estimators=16, max_depth=3),
#  'vote':    lambda: VotingClassifier(),
#  'bag':     lambda: BaggingClassifier(),
  'adabst':  lambda: AdaBoostClassifier(n_estimators=16),
  'gbst':    lambda: GradientBoostingClassifier(n_estimators=16, max_depth=3),
  'hgbst':   lambda: HistGradientBoostingClassifier(max_depth=3),
  'hgbst':   lambda: BernoulliNB(),
  'gnb':     lambda: GaussianNB(),
  'mnb':     lambda: MultinomialNB(),
  'cnb':     lambda: ComplementNB(),
#  'catnb':   lambda: CategoricalNB(),
  'ridge':   lambda: RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1]),
  'log':     lambda: LogisticRegressionCV(),
  'paclf':   lambda: PassiveAggressiveClassifier(),
  'sgdclf':  lambda: SGDClassifier(),
  'svc-l':   lambda: SVC(kernel='linear'),
  'svc-p':   lambda: SVC(kernel='poly'),
  'svc-r':   lambda: SVC(kernel='rbf'),
  'svc-s':   lambda: SVC(kernel='sigmoid'),
  'nusvc-l': lambda: NuSVC(kernel='linear'),
  'nusvc-p': lambda: NuSVC(kernel='poly'),
  'nusvc-r': lambda: NuSVC(kernel='rbf'),
  'nusvc-s': lambda: NuSVC(kernel='sigmoid'),
  'mlp-d64-s50':  lambda: MLPClassifier(hidden_layer_sizes=[64], max_iter=50),
  'mlp-d64-s100': lambda: MLPClassifier(hidden_layer_sizes=[64], max_iter=100),
  'mlp-d16-s50':  lambda: MLPClassifier(hidden_layer_sizes=[16], max_iter=50),
  'mlp-d16-s100': lambda: MLPClassifier(hidden_layer_sizes=[16], max_iter=100),
#  'rbm':     lambda: BernoulliRBM(n_components=256),
}


def run_tfidf(analyzer:str) -> Datasets:
  ''' This should be more like syntaxical feature '''
  assert analyzer in ['char', 'word'] or analyzer.endswith('gram')

  def process_data(split:str, tfidfvec:TfidfVectorizer) -> Tuple[NDArray, NDArray]:
    T, Y = load_dataset(split)
    tfidf = tfidfvec.fit_transform(T) if split == 'train' else tfidfvec.transform(T)
    X = tfidf.todense()     # [N=1600, K=3386]
    if isinstance(X, np.matrix): X = X.A
    return X, Y

  if analyzer == 'char':
    tokenizer = None
    stop_words = STOP_WORDS_CHAR
  elif analyzer == 'word':
    tokenizer = jieba.lcut_for_search
    stop_words = STOP_WORDS_WORD
  elif analyzer.endswith('gram'):
    tokenizer = make_tokenizer(LOG_PATH / analyzer / 'vocab.txt')
    stop_words = None
    analyzer = 'word'   # NOTE: overrides

  tfidfvec = TfidfVectorizer(analyzer=analyzer, tokenizer=tokenizer, stop_words=stop_words)
  X_train, Y_train = process_data('train', tfidfvec)
  X_test,  Y_test  = process_data('test',  tfidfvec)
  X_valid, Y_valid = process_data('valid', tfidfvec)
  return (X_train, Y_train), (X_test,  Y_test), (X_valid, Y_valid)


def run_fasttext(analyzer:str) -> Datasets:
  ''' This should be more like semantical feature '''
  assert analyzer in ['char', 'word', 'sent'] or analyzer.endswith('gram')

  if not FASTTEXT_CKPT_PATH.exists():
    import shutil
    fasttext.util.download_model('zh', if_exists='ignore')
    BASE_PATH = Path(__file__).absolute()
    shutil.move(BASE_PATH / 'cc.zh.300.bin',    DATA_PATH)
    shutil.move(BASE_PATH / 'cc.zh.300.bin.gz', DATA_PATH)
  embed: FastText = fasttext.load_model(str(FASTTEXT_CKPT_PATH))

  def process_data(split:str) -> Dataset:
    T, Y = load_dataset(split)
    if analyzer == 'char':
      X = [np.stack([embed.get_word_vector(w) for w in list(t) if w in embed and w not in STOP_WORDS_CHAR], axis=0).mean(axis=0) for t in T]
    elif analyzer == 'word':
      X = [np.stack([embed.get_word_vector(w) for w in jieba.cut_for_search(t) if w in embed and w not in STOP_WORDS_WORD], axis=0).mean(axis=0) for t in T]
    elif analyzer == 'sent':
      X = [embed.get_sentence_vector(t) for t in T]
    elif analyzer.endswith('gram'):
      tokenizer = make_tokenizer(LOG_PATH / analyzer / 'vocab.txt')
      X = [np.stack([(embed.get_word_vector(w) if w in embed else embed.get_sentence_vector(w)) for w in tokenizer(t)], axis=0).mean(axis=0) for t in T]
    return np.stack(X, axis=0), Y

  X_train, Y_train = process_data('train')
  X_test,  Y_test  = process_data('test')
  X_valid, Y_valid = process_data('valid')
  return (X_train, Y_train), (X_test,  Y_test), (X_valid, Y_valid)


def run_visualize(datasets:Datasets, name:str, out_dp:Path):
  (X_train, Y_train), (X_test, Y_test), (X_valid, Y_valid) = datasets

  def save_plot(fp:Path, z_train, z_test, z_valid, s=1):
    nonlocal Y_train, Y_test, Y_valid

    plt.subplot(221)
    plt.scatter(z_train[:, 0], z_train[:, 1], s, c=Y_train, marker='o', alpha=0.7, label='train')
    plt.scatter(z_test [:, 0], z_test [:, 1], s, c=Y_test,  marker='x', alpha=0.7, label='test')
    plt.scatter(z_valid[:, 0], z_valid[:, 1], s, c=Y_valid, marker='*', alpha=0.7, label='valid')
    plt.axis('off')
    plt.title('all')

    plt.subplot(222)
    plt.scatter(z_valid[:, 0], z_valid[:, 1], s, c=Y_valid, marker='*', alpha=0.7, label='valid')
    plt.axis('off')
    plt.title('valid')

    plt.subplot(223)
    plt.scatter(z_train[:, 0], z_train[:, 1], s, c=Y_train, marker='o', alpha=0.7, label='train')
    plt.axis('off')
    plt.title('train')
    
    plt.subplot(224)
    plt.scatter(z_test [:, 0], z_test [:, 1], s, c=Y_test,  marker='x', alpha=0.7, label='test')
    plt.axis('off')
    plt.title('test')
    
    plt.suptitle(fp.stem)
    plt.tight_layout()
    plt.subplots_adjust()
    plt.savefig(fp, dpi=600)

  if 'pca':
    pca = PCA(n_components=2)
    z_train = pca.fit_transform(X_train)
    z_test  = pca.    transform(X_test)
    z_valid = pca.    transform(X_valid)

    save_plot(out_dp / f'pca_{name}.png', z_train, z_test, z_valid)

  if 'kpca':    # NOTE: just alike pca, not giving anything new
    for k in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']:
      kpca = KernelPCA(n_components=2, kernel=k)
      z_train = kpca.fit_transform(X_train)
      z_test  = kpca.    transform(X_test)
      z_valid = kpca.    transform(X_valid)

      save_plot(out_dp / f'kpca-{k}_{name}.png', z_train, z_test, z_valid)

  if 'tsne':
    tsne = TSNE(n_components=2)
    z_all = tsne.fit_transform(np.concatenate([X_train, X_test, X_valid], axis=0))
    cp  = len(X_train)
    cp2 = cp + len(X_test)

    save_plot(out_dp / f'tsne_{name}.png', z_all[:cp, :], z_all[cp:cp2, :], z_all[cp2:, :])


def run_model(name, model, datasets:Datasets, logger:Logger) -> Scores:
  (X_train, Y_train), (X_test, Y_test), (X_valid, Y_valid) = datasets

  model.fit(X_train, Y_train)

  precs, recalls, f1s, cmats = [], [], [], []

  logger.info(f'[{name}]')
  for split in SPLITS:
    logger.info(f'<{split}>')

    X_split = locals().get(f'X_{split}')
    Y_split = locals().get(f'Y_{split}')
    Y_pred = model.predict(X_split)

    prec, recall, f1, _ = precision_recall_fscore_support(Y_split, Y_pred, average=None)
    cmat = confusion_matrix(Y_split, Y_pred)
    precs  .append(prec)
    recalls.append(recall)
    f1s    .append(f1)
    cmats  .append(cmat)

    logger.info(f'>> prec: {prec}')
    logger.info(f'>> recall: {recall}')
    logger.info(f'>> f1: {f1}')
    logger.info('>> confusion_matrix:')
    logger.info(cmat)
  
    logger.info('-' * 32)
  logger.info('\n')

  return precs, recalls, f1s, cmats


def go_train(args):
  out_dp: Path = LOG_PATH / args.analyzer / args.feature
  out_dp.mkdir(exist_ok=True, parents=True)

  datasets: Datasets = globals()[f'run_{args.feature}'](args.analyzer)
  run_visualize(datasets, f'{args.feature}-{args.analyzer}', out_dp)

  logger = get_logger(out_dp / 'run.log', mode='w')
  result = { }
  for name, model_fn in MODELS.items():
    print(f'<< running {name}...')
    try:
      logger.info(f'exp: {args.feature}-{args.analyzer}-{name}')
      precs, recalls, f1s, cmats = run_model(name, model_fn(), datasets, logger)
      result[name] = {
        'prec':   precs,
        'recall': recalls,
        'f1':     f1s,
        'cmat':   cmats,
      }
    except: print_exc()

  with open(out_dp / 'result.pkl', 'wb') as fh:
    pkl.dump(result, fh)


def go_eval(args):
  for feature in FEATURES:
    for analyzer in ANALYZERS:
      out_dp = LOG_PATH / analyzer / feature
      if not out_dp.exists(): continue

      with open(out_dp / 'result.pkl', 'rb') as fh:
        result = pkl.load(fh)

      names, f1s = [], []
      for name, scores in result.items():
        names.append(name)
        f1s.append(np.stack(scores['f1'], axis=-1))
      f1s = np.stack(f1s, axis=0)     # [n_model=30, n_cls=4, n_split=3]

      plt.clf()
      plt.figure(figsize=(6, 8))
      n_fig = f1s.shape[-1]
      for i in range(n_fig):
        plt.subplot(n_fig, 1, i+1)
        for j in range(f1s.shape[-2]):    # each class
          plt.plot(f1s[:, j, i], label=j)
        plt.title(SPLITS[i])
        plt.legend(loc=4, prop={'size': 6})
        plt.xticks(ticks=range(len(names)), labels=names, rotation=90, ha='right')
      plt.suptitle(f'f1-score {feature}-{analyzer}')
      plt.tight_layout()
      plt.savefig(LOG_PATH / f'scores_{feature}_{analyzer}.png', dpi=600)

      plt.clf()
      plt.figure(figsize=(6, 8))
      n_fig = f1s.shape[-1]
      for i in range(n_fig):
        plt.subplot(n_fig, 1, i+1)
        plt.plot(f1s[:, :, i].mean(axis=1), label=j)
        plt.title(SPLITS[i])
        plt.xticks(ticks=range(len(names)), labels=names, rotation=90, ha='right')
      plt.suptitle(f'f1-score avg. {feature}-{analyzer}')
      plt.tight_layout()
      plt.savefig(LOG_PATH / f'scores_{feature}_{analyzer}-avg.png', dpi=600)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-L', '--analyzer', choices=ANALYZERS, help='tokenize level')
  parser.add_argument('-F', '--feature',  choices=FEATURES, help='input feature')
  parser.add_argument('--eval', action='store_true', help='compare result scores')
  args = parser.parse_args()

  if args.eval:
    go_eval()
    exit(0)

  go_train()
