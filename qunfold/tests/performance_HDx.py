import argparse
import numpy as np
import time
from qunfold import HistogramTransformer
from qunfold.tests import RNG, make_problem, generate_data
from scipy.sparse import csr_matrix

class JaxHistogramTransformer(HistogramTransformer):
  """This implementation of the HistogramTransformer is based on JAX's JIT capabilities."""
  pass

class SparseHistogramTransformer(HistogramTransformer):
  """This implementation of the HistogramTransformer is based on SciPy's sparse API."""
  def _transform_after_preprocessor(self, X):
    fX = []
    for j in range(X.shape[1]): # feature index
      e = self.edges[j]
      i_row = np.arange(X.shape[0])
      i_col = np.clip(np.ceil((X[:,j] - e[0]) / (e[1]-e[0])).astype(int), 0, self.n_bins-1)
      fX_j = csr_matrix(
        (np.ones(X.shape[0], dtype=int), (i_row, i_col)),
        shape = (X.shape[0], self.n_bins),
      )
      fX.append(fX_j.toarray())
    fX = np.stack(fX).swapaxes(0, 1).reshape((X.shape[0], -1))
    if self.unit_scale:
      fX = fX / fX.sum(axis=1, keepdims=True)
    return fX

class NaiveHistogramTransformer(HistogramTransformer):
  """This implementation of the HistogramTransformer is based on for loops."""
  def fit_transform(self, X, y):
    if y.min() not in [0, 1]:
      raise ValueError("y.min() ∉ [0, 1]")
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y)
      self.n_classes = self.preprocessor.n_classes
    else:
      y -= y.min() # map to zero-based labels
      self.n_classes = len(np.unique(y))
    self.edges = []
    for x in X.T: # iterate over columns = features
      e = np.histogram_bin_edges(x, bins=self.n_bins)[1:]
      e[-1] = np.inf # THIS LINE IS DIFFERENT FROM THE ORIGINAL fit_transform
      self.edges.append(e)
    return self._transform_after_preprocessor(X), y
  def _transform_after_preprocessor(self, X):
    fX = np.zeros((X.shape[0], self.n_bins * X.shape[1]), dtype=int)
    for j in range(X.shape[1]): # feature index
      e = self.edges[j]
      offset = j * self.n_bins
      for i in range(X.shape[0]): # sample index
        fX[i, offset + np.argmax(e >= X[i,j])] = 1 # argmax returns the index of the 1st True
    if self.unit_scale:
      fX = fX / fX.sum(axis=1, keepdims=True)
    return fX

def main():
  transformers = [
    HistogramTransformer,
    JaxHistogramTransformer,
    SparseHistogramTransformer,
    NaiveHistogramTransformer,
  ]
  results = {} # mapping from class names to prediction times
  n_unequal = 0
  for _ in range(10):
    q, M, p_trn = make_problem()
    X_trn, y_trn = generate_data(M, p_trn)
    p_tst = RNG.permutation(p_trn)
    X_tst, y_tst = generate_data(M, p_tst)
    for n_bins in [2, 4, 8]:
      fX_ref = None
      for i, transformer in enumerate(transformers):
        t = transformer(n_bins)
        t.fit_transform(X_trn, y_trn)
        start = time.time()
        fX = t.transform(X_tst)
        name = transformer.__name__
        times = results.get(name, [])
        times.append(time.time() - start)
        results[name] = times
        if fX_ref is None:
          fX_ref = fX
        elif not np.all(fX == fX_ref):
          print(name)
          print(fX_ref[:5,:2])
          print(fX[:5,:2])
          print(X_tst[:5,0])
          print()
          n_unequal += 1
  for name, times in results.items():
    print(f"{name}: {np.mean(times)}")
  print(f"{n_unequal} unequal results between methods")

if __name__ == '__main__':
  main()
