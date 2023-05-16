#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/05/10 

from run_baseline_sk import *

# data visualization through PCA/TSNE projection

METHODS = {
  'pca':    lambda: PCA      (n_components=3),
  'kpca-l': lambda: KernelPCA(n_components=3, kernel='linear'),
  'kpca-p': lambda: KernelPCA(n_components=3, kernel='poly'),
  'kpca-r': lambda: KernelPCA(n_components=3, kernel='rbf'),
  'kpca-s': lambda: KernelPCA(n_components=3, kernel='sigmoid'),
  'kpca-c': lambda: KernelPCA(n_components=3, kernel='cosine'),
  'tsne':   lambda: TSNE     (n_components=3)
}


def plot_project(model:Union[PCA, TSNE], datasets:Datasets, name:str):
  (X_train, Y_train), (X_test, Y_test), (X_valid, Y_valid) = datasets

  if isinstance(model, TSNE):
    z_all = model.fit_transform(np.concatenate([X_train, X_test, X_valid], axis=0))
    cp  =      len(X_train)
    cp2 = cp + len(X_test)

    z_train = z_all[:cp,    :]
    z_test  = z_all[cp:cp2, :]
    z_valid = z_all[cp2:,   :]
  else:
    z_train = model.fit_transform(X_train)
    z_test  = model.    transform(X_test)
    z_valid = model.    transform(X_valid)

  plt.clf()
  ax = plt.axes(projection='3d')
  ax.scatter(z_train[:, 0], z_train[:, 1], z_train[:, 2], zdir='z', s=10, c=Y_train, marker='o', alpha=0.7, label='train')
  ax.scatter(z_test [:, 0], z_test [:, 1], z_test [:, 2], zdir='z', s=10, c=Y_test,  marker='x', alpha=0.7, label='test')
  ax.scatter(z_valid[:, 0], z_valid[:, 1], z_valid[:, 2], zdir='z', s=10, c=Y_valid, marker='*', alpha=0.7, label='valid')
  plt.suptitle(name)
  plt.tight_layout()
  plt.legend()
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--method',   default='pca',   choices=METHODS.keys(), help='project method')
  parser.add_argument('-L', '--analyzer', default='char',  choices=ANALYZERS,      help='tokenize level')
  parser.add_argument('-F', '--feature',  default='tfidf', choices=FEATURES,       help='input feature')
  args = parser.parse_args()

  model = METHODS[args.method]()
  datasets: Datasets = globals()[f'run_{args.feature}'](args.analyzer)
  plot_project(model, datasets, f'{args.analyzer}-{args.feature}')
