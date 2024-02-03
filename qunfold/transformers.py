import numpy as np
import warnings
from abc import ABC, abstractmethod
from functools import partial
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

def class_prevalences(y, n_classes=None):
  """Determine the prevalence of each class.

  Args:
      y: An array of labels, shape (n_samples,).
      n_classes (optional): The number of classes. Defaults to `None`, which corresponds to `np.max(y)+1`.

  Returns:
      An array of class prevalences that sums to one, shape (n_classes,).
  """
  if n_classes is None:
    n_classes = np.max(y)+1
  n_samples_per_class = np.zeros(n_classes, dtype=int)
  i, n = np.unique(y, return_counts=True)
  n_samples_per_class[i] = n # non-existing classes maintain a zero entry
  return n_samples_per_class / n_samples_per_class.sum() # normalize to prevalences

# helper function for ensuring sane labels
def _check_y(y, n_classes=None):
  if n_classes is not None:
    if n_classes != np.max(y)+1:
      warnings.warn(f"Classes are missing: n_classes != np.max(y)+1 = {np.max(y)+1}")

class AbstractTransformer(ABC):
  """Abstract base class for transformers."""
  @abstractmethod
  def fit_transform(self, X, y, average=True, n_classes=None):
    """This abstract method has to fit the transformer and to return the transformation of the input data.

    Note:
        Implementations of this abstract method should check the sanity of labels by calling `_check_y(y, n_classes)` and they must set the property `self.p_trn = class_prevalences(y, n_classes)`.

    Args:
        X: The feature matrix to which this transformer will be fitted.
        y: The labels to which this transformer will be fitted.
        average (optional): Whether to return a transfer matrix `M` or a transformation `(f(X), y)`. Defaults to `True`.
        n_classes (optional): The number of expected classes. Defaults to `None`.

    Returns:
        A transfer matrix `M` if `average==True` or a transformation `(f(X), y)` if `average==False`.
    """
    pass
  @abstractmethod
  def transform(self, X, average=True):
    """This abstract method has to transform the data `X`.

    Args:
        X: The feature matrix that will be transformed.
        average (optional): Whether to return a vector `q` or a transformation `f(X)`. Defaults to `True`.

    Returns:
        A vector `q = f(X).mean(axis=0)` if `average==True` or a transformation `f(X)` if `average==False`.
    """
    pass

# helper function for ClassTransformer(..., is_probabilistic=False)
def _onehot_encoding(y, n_classes):
  return np.eye(n_classes)[y] # https://stackoverflow.com/a/42874726/20580159

class ClassTransformer(AbstractTransformer):
  """A classification-based feature transformation.

  This transformation can either be probabilistic (using the posterior predictions of a classifier) or crisp (using the class predictions of a classifier). It is used in ACC, PACC, CC, PCC, and SLD.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      is_probabilistic (optional): Whether probabilistic or crisp predictions of the `classifier` are used to transform the data. Defaults to `False`.
      fit_classifier (optional): Whether to fit the `classifier` when this transformer is fitted. Defaults to `True`.
  """
  def __init__(self, classifier, is_probabilistic=False, fit_classifier=True):
    self.classifier = classifier
    self.is_probabilistic = is_probabilistic
    self.fit_classifier = fit_classifier
  def fit_transform(self, X, y, average=True, n_classes=None):
    if not hasattr(self.classifier, "oob_score") or not self.classifier.oob_score:
      raise ValueError(
        "The ClassTransformer either requires a bagging classifier with oob_score=True",
        "or an instance of qunfold.sklearn.CVClassifier"
      )
    _check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    if self.fit_classifier:
      self.classifier.fit(X, y)
    fX = np.zeros((len(X), n_classes))
    fX[:, self.classifier.classes_] = self.classifier.oob_decision_function_
    is_finite = np.all(np.isfinite(fX), axis=1)
    fX = fX[is_finite,:] # drop instances that never became OOB
    y = y[is_finite]
    if not self.is_probabilistic:
      fX = _onehot_encoding(np.argmax(fX, axis=1), n_classes)
    if average:
      M = np.zeros((n_classes, n_classes))
      for c in range(n_classes):
        if np.sum(y==c) > 0:
          M[:,c] = fX[y==c].mean(axis=0)
      return M
    return fX, y
  def transform(self, X, average=True):
    n_classes = len(self.p_trn)
    fX = np.zeros((len(X), n_classes))
    fX[:, self.classifier.classes_] = self.classifier.predict_proba(X)
    if not self.is_probabilistic:
      fX = _onehot_encoding(np.argmax(fX, axis=1), n_classes)
    if average:
      return fX.mean(axis=0) # = q
    return fX

class DistanceTransformer(AbstractTransformer):
  """A distance-based feature transformation, as it is used in `EDx` and `EDy`.

  Args:
      metric (optional): The metric with which the distance between data items is measured. Can take any value that is accepted by `scipy.spatial.distance.cdist`. Defaults to `"euclidean"`.
      preprocessor (optional): Another `AbstractTransformer` that is called before this transformer. Defaults to `None`.
  """
  def __init__(self, metric="euclidean", preprocessor=None):
    self.metric = metric
    self.preprocessor = preprocessor
  def fit_transform(self, X, y, average=True, n_classes=None):
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, average=False, n_classes=n_classes)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      _check_y(y, n_classes)
      self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    self.X_trn = X
    self.y_trn = y
    if average:
      M = np.zeros((n_classes, n_classes))
      for c in range(n_classes):
        if np.sum(y==c) > 0:
          M[:,c] = self._transform_after_preprocessor(X[y==c])
      return M
    else:
      return self._transform_after_preprocessor(X, average=False), y
  def transform(self, X, average=True):
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    return self._transform_after_preprocessor(X, average=average)
  def _transform_after_preprocessor(self, X, average=True):
    n_classes = len(self.p_trn)
    fX = np.zeros((X.shape[0], n_classes))
    for c in range(n_classes):
      if np.sum(self.y_trn==c) > 0:
        fX[:,c] = cdist(X, self.X_trn[self.y_trn==c], metric=self.metric).mean(axis=1)
    if average:
      return fX.mean(axis=0) # = q
    return fX

class HistogramTransformer(AbstractTransformer):
  """A histogram-based feature transformation, as it is used in `HDx` and `HDy`.

  Args:
      n_bins: The number of bins in each feature.
      preprocessor (optional): Another `AbstractTransformer` that is called before this transformer. Defaults to `None`.
      unit_scale (optional): Whether or not to scale each output to a sum of one. A value of `False` indicates that the sum of each output is the number of features. Defaults to `True`.
  """
  def __init__(self, n_bins, preprocessor=None, unit_scale=True):
    self.n_bins = n_bins
    self.preprocessor = preprocessor
    self.unit_scale = unit_scale
  def fit_transform(self, X, y, average=True, n_classes=None):
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, average=False, n_classes=n_classes)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      _check_y(y, n_classes)
      self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    self.edges = []
    for x in X.T: # iterate over columns = features
      e = np.histogram_bin_edges(x, bins=self.n_bins)
      self.edges.append(e)
    if average:
      M = np.zeros((X.shape[1] * self.n_bins, n_classes))
      for c in range(n_classes):
        if np.sum(y==c) > 0:
          M[:,c] = self._transform_after_preprocessor(X[y==c])
      return M
    return self._transform_after_preprocessor(X, average=average), y
  def transform(self, X, average=True):
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    return self._transform_after_preprocessor(X, average=average)
  def _transform_after_preprocessor(self, X, average=True):
    if not average:
      fX = []
      for j in range(X.shape[1]): # feature index
        e = self.edges[j][1:]
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
    else: # a concatenation of numpy histograms is faster to compute
      histograms = []
      for j in range(X.shape[1]):  # feature index
        e = np.copy(self.edges[j])
        e[0] = -np.inf # always use exactly self.n_bins and never omit any items
        e[-1] = np.inf
        hist, _ = np.histogram(X[:, j], bins=e)
        if self.unit_scale:
          hist = hist / X.shape[1]
        histograms.append(hist / X.shape[0])
      return np.concatenate(histograms) # = q

class EnergyKernelTransformer(AbstractTransformer):
  """A kernel-based feature transformation, as it is used in `KMM`, that uses the `energy` kernel:

      k(x_1, x_2) = ||x_1|| + ||x_2|| - ||x_1 - x_2||

  Note:
      The methods of this transformer do not support setting `average=False`.

  Args:
      preprocessor (optional): Another `AbstractTransformer` that is called before this transformer. Defaults to `None`.
  """
  def __init__(self, preprocessor=None):
    self.preprocessor = preprocessor
  def fit_transform(self, X, y, average=True, n_classes=None):
    if not average:
      raise ValueError("EnergyKernelTransformer does not support average=False")
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, average=False, n_classes=n_classes)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      _check_y(y, n_classes)
      self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    self.X_trn = X
    self.y_trn = y
    self.norms = np.zeros(n_classes) # = ||x||
    for c in range(len(self.norms)):
      if np.sum(y==c) > 0:
        self.norms[c] = np.linalg.norm(X[y==c], axis=1).mean()
    M = np.zeros((n_classes, n_classes))
    for c in range(n_classes):
      if np.sum(y==c) > 0:
        M[:,c] = self._transform_after_preprocessor(X[y==c], self.norms[c])
    return M
  def transform(self, X, average=True):
    if not average:
      raise ValueError("EnergyKernelTransformer does not support average=False")
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    norm = np.linalg.norm(X, axis=1).mean()
    return self._transform_after_preprocessor(X, norm)
  def _transform_after_preprocessor(self, X, norm):
    n_classes = len(self.p_trn)
    dists = np.zeros(n_classes) # = ||x_1 - x_2|| for all x_2 = X_trn[y_trn == i]
    for c in range(n_classes):
      if np.sum(self.y_trn==c) > 0:
        dists[c] = cdist(X, self.X_trn[self.y_trn==c], metric="euclidean").mean()
    return norm + self.norms - dists # = ||x_1|| + ||x_2|| - ||x_1 - x_2|| for all x_2

class GaussianKernelTransformer(AbstractTransformer):
  """A kernel-based feature transformation, as it is used in `KMM`, that uses the `gaussian` kernel:

      k(x, y) = exp(-||x - y||^2 / (2σ^2))

  Args:
      sigma (optional): A smoothing parameter of the kernel function. Defaults to `1`.
      preprocessor (optional): Another `AbstractTransformer` that is called before this transformer. Defaults to `None`.
  """
  def __init__(self, sigma=1, preprocessor=None):
    self.sigma = sigma
    self.preprocessor = preprocessor
  def fit_transform(self, X, y, average=True, n_classes=None):
    if not average:
      raise ValueError("GaussianKernelTransformer does not support average=False")
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, average=False, n_classes=n_classes)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      _check_y(y, n_classes)
      self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    self.X_trn = X
    self.y_trn = y
    M = np.zeros((n_classes, n_classes))
    for c in range(n_classes):
      M[:,c] = self._transform_after_preprocessor(X[y==c])
    return M
  def transform(self, X, average=True):
    if not average:
      raise ValueError("GaussianKernelTransformer does not support average=False")
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    return self._transform_after_preprocessor(X)
  def _transform_after_preprocessor(self, X):
    n_classes = len(self.p_trn)
    res = np.zeros(n_classes)
    for i in range(n_classes):
      norm_fac = X.shape[0] * self.X_trn[self.y_trn==i].shape[0]
      sq_dists = cdist(X, self.X_trn[self.y_trn == i], metric="euclidean")**2
      res[i] = np.exp(-sq_dists / 2*self.sigma**2).sum() / norm_fac
    return res

class KernelTransformer(AbstractTransformer):
  """A general kernel-based feature transformation, as it is used in `KMM`. If you intend to use a Gaussian kernel or energy kernel, prefer their dedicated and more efficient implementations over this class.

  Note:
      The methods of this transformer do not support setting `average=False`.

  Args:
      kernel: A callable that will be used as the kernel. Must follow the signature `(X[y==i], X[y==j]) -> scalar`.
  """
  def __init__(self, kernel):
    self.kernel = kernel
  def fit_transform(self, X, y, average=True, n_classes=None):
    if not average:
      raise ValueError("KernelTransformer does not support average=False")
    _check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    self.X_trn = X
    self.y_trn = y
    M = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
      for j in range(i, n_classes):
        if np.sum(y==i) > 0 and np.sum(y==j):
          M[i, j] = self.kernel(X[y==i], X[y==j])
          if i != j:
            M[j, i] = M[i, j]
    return M
  def transform(self, X, average=True):
    if not average:
      raise ValueError("KernelTransformer does not support average=False")
    n_classes = len(self.p_trn)
    q = np.zeros(n_classes)
    for c in range(n_classes):
      if np.sum(self.y_trn==c) > 0:
        q[c] = self.kernel(self.X_trn[self.y_trn==c], X)
    return q

# kernel function for the LaplacianKernelTransformer
def _laplacianKernel(X, Y, sigma):
    nx = X.shape[0]
    ny = Y.shape[0]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    Y = Y.reshape((1, Y.shape[0], Y.shape[1]))
    D_lk = np.abs(((X - Y))).sum(-1) 
    K_ij = np.exp((-sigma * D_lk)).sum(0).sum(0) / (nx * ny)
    return K_ij

class LaplacianKernelTransformer(KernelTransformer):
  """A kernel-based feature transformation, as it is used in `KMM`, that uses the `laplacian` kernel.

  Args:
      sigma (optional): A smoothing parameter of the kernel function. Defaults to `1`.
  """
  def __init__(self, sigma=1):
    self.sigma = sigma
  @property # implement self.kernel as a property to allow for hyper-parameter tuning of sigma
  def kernel(self):
    return partial(_laplacianKernel, sigma=self.sigma)

class GaussianRFFKernelTransformer(AbstractTransformer):
  """An efficient approximation of the `GaussianKernelTransformer`, as it is used in `KMM`, using random Fourier features.

  Args:
      sigma (optional): A smoothing parameter of the kernel function. Defaults to `1`.
      n_rff (optional): The number of random Fourier features. Defaults to `1000`.
      preprocessor (optional): Another `AbstractTransformer` that is called before this transformer. Defaults to `None`.
      seed (optional): Controls the randomness of the random Fourier features. Defaults to `None`.
  """
  def __init__(self, sigma=1, n_rff=1000, preprocessor=None, seed=None):
    self.sigma = sigma
    self.n_rff = n_rff
    self.preprocessor = preprocessor
    self.seed = seed
  def fit_transform(self, X, y, average=True, n_classes=None):
    if not average:
      raise ValueError("GaussianRFFKernelTransformer does not support average=False")
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, average=False, n_classes=n_classes)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      _check_y(y, n_classes)
      self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    self.X_trn = X
    self.y_trn = y
    self.w = np.random.default_rng(self.seed).normal(
      loc = 0,
      scale = (1. / self.sigma),
      size = (self.n_rff // 2, X.shape[1]),
    ).astype(np.float32)
    self.mu = np.stack(
      [ self._transform_after_preprocessor(X[y==c]) for c in range(n_classes) ],
      axis = 1
    )
    self.M = self.mu.T @ self.mu
    return self.M
  def transform(self, X, average=True):
    if not average:
      raise ValueError("GaussianRFFKernelTransformer does not support average=False")
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    return self._transform_after_preprocessor(X) @ self.mu
  def _transform_after_preprocessor(self, X):
    Xw = X @ self.w.T
    C = np.concatenate((np.cos(Xw), np.sin(Xw)), axis=1)
    return np.sqrt(2 / self.n_rff) * np.mean(C, axis=0)
