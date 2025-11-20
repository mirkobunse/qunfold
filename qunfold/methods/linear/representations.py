import numpy as np
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from typing import Any, Callable, Optional
from .. import class_prevalences, check_y
from ...base import BaseMixin

class AbstractRepresentation(ABC,BaseMixin):
  """Abstract base class for representations."""
  @abstractmethod
  def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
    """This abstract method has to fit the representation and to return the transformed input data.

    Note:
        Implementations of this abstract method should check the sanity of labels by calling `check_y(y, n_classes)` and they must set the property `self.p_trn = class_prevalences(y, n_classes)`.

    Args:
        X: The feature matrix to which this representation will be fitted.
        y: The labels to which this representation will be fitted.
        sample_weight (optional): Importance weights for each (X[i], y[i]) pair to use during fitting. Defaults to `None`.
        average (optional): Whether to return a transfer matrix `M` or a transformation `(f(X), y)`. Defaults to `True`.
        n_classes (optional): The number of expected classes. Defaults to `None`.

    Returns:
        A transfer matrix `M` if `average==True` or a transformation `(f(X), y)` if `average==False`.
    """
    pass
  @abstractmethod
  def transform(self, X, sample_weight=None, average=True):
    """This abstract method has to transform the data `X`.

    Args:
        X: The feature matrix that will be transformed.
        sample_weight (optional): Importance weights for each X[i] to use during averaging if `average==True`. Defaults to `None`.
        average (optional): Whether to return a vector `q` or a transformation `f(X)`. Defaults to `True`.

    Returns:
        A vector `q = f(X).average(axis=0, weights=sample_weight)` if `average==True` or a transformation `f(X)` if `average==False`.
    """
    pass

# helper function for ClassRepresentation(..., is_probabilistic=False)
def _onehot_encoding(y, n_classes):
  return np.eye(n_classes)[y] # https://stackoverflow.com/a/42874726/20580159

@dataclass
class ClassRepresentation(AbstractRepresentation):
  """A classification-based data representation.

  This representation can either be probabilistic (using the posterior predictions of a classifier) or crisp (using the class predictions of a classifier). It is used in ACC, PACC, CC, PCC, and SLD.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      is_probabilistic (optional): Whether probabilistic or crisp predictions of the `classifier` are used to represent the data. Defaults to `False`.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
  """
  classifier: Any
  is_probabilistic: bool = False
  fit_classifier: bool = True
  def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
    if not hasattr(self.classifier, "oob_score") or not self.classifier.oob_score:
      raise ValueError(
        "The ClassRepresentation either requires a bagging classifier with oob_score=True",
        "or an instance of qunfold.sklearn.CVClassifier"
      )
    check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    if self.fit_classifier:
      try:
        self.classifier.fit(X, y, sample_weight=sample_weight)
      except TypeError:
        self.classifier.fit(X, y)

    fX = np.zeros((X.shape[0], n_classes))
    fX[:, self.classifier.classes_] = self.classifier.oob_decision_function_
    is_finite = np.all(np.isfinite(fX), axis=1)
    fX = fX[is_finite,:] # drop instances that never became OOB
    y = y[is_finite]
    if sample_weight is not None:
      sample_weight = sample_weight[is_finite]
    if not self.is_probabilistic:
      fX = _onehot_encoding(np.argmax(fX, axis=1), n_classes)
    if average:
      M = np.zeros((n_classes, n_classes))
      for c in range(n_classes):
        if np.sum(y==c) > 0:
          weights = None
          if sample_weight is not None:
            weights = sample_weight[y==c]
          M[:,c] = np.average(fX[y==c], axis=0, weights=weights)
      return M
    return fX, y
  def transform(self, X, sample_weight=None, average=True): # TODO: unit_scale not needed here?
    n_classes = len(self.p_trn)
    fX = np.zeros((X.shape[0], n_classes))
    fX[:, self.classifier.classes_] = self.classifier.predict_proba(X)
    if not self.is_probabilistic:
      fX = _onehot_encoding(np.argmax(fX, axis=1), n_classes)
    if average:
      return np.average(fX, axis=0, weights=sample_weight) # = q
    return fX

@dataclass
class DistanceRepresentation(AbstractRepresentation):
  """A distance-based data representation, as it is used in `EDx` and `EDy`.

  Args:
      metric (optional): The metric with which the distance between data items is measured. Can take any value that is accepted by `scipy.spatial.distance.cdist`. Defaults to `"euclidean"`.
      preprocessor (optional): Another `AbstractRepresentation` that is called before this representation. Defaults to `None`.
  """
  metric: str = "euclidean"
  preprocessor: Optional[AbstractRepresentation] = None
  def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, sample_weight=sample_weight, average=False, n_classes=n_classes)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      check_y(y, n_classes)
      self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    self.X_trn = X
    self.y_trn = y
    if average:
      M = np.zeros((n_classes, n_classes))
      for c in range(n_classes):
        if np.sum(y==c) > 0:
          weights = None
          if sample_weight is not None:
            weights = sample_weight[y==c]
          M[:,c] = self._transform_after_preprocessor(X[y==c], sample_weight=weights)
      return M
    else:
      return self._transform_after_preprocessor(X, average=False), y
  def transform(self, X, sample_weight=None, average=True):
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    return self._transform_after_preprocessor(
      X, sample_weight=sample_weight, average=average)
  def _transform_after_preprocessor(self, X, sample_weight=None, average=True):
    n_classes = len(self.p_trn)
    fX = np.zeros((X.shape[0], n_classes))
    for c in range(n_classes):
      if np.sum(self.y_trn==c) > 0:
        fX[:,c] = cdist(X, self.X_trn[self.y_trn==c], metric=self.metric).mean(axis=1)
    if average:
      return np.average(fX, axis=0, weights=sample_weight) # = q
    return fX

@dataclass
class HistogramRepresentation(AbstractRepresentation):
  """A histogram-based data representation, as it is used in `HDx` and `HDy`.

  Args:
      n_bins: The number of bins in each feature.
      preprocessor (optional): Another `AbstractRepresentation` that is called before this representation. Defaults to `None`.
      unit_scale (optional): Whether or not to scale each output to a sum of one. A value of `False` indicates that the sum of each output is the number of features. Defaults to `True`.
  """
  n_bins: int
  preprocessor: Optional[AbstractRepresentation] = None
  unit_scale: bool = True
  def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, sample_weight=sample_weight, average=False, n_classes=n_classes)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      check_y(y, n_classes)
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
          weights = None
          if sample_weight is not None:
            weights = sample_weight[y==c]
          M[:,c] = self._transform_after_preprocessor(X[y==c], sample_weight=weights)
      return M
    fX = self._transform_after_preprocessor(
      X,
      average = average,
      sample_weight = sample_weight,
    )
    return fX, y
  def transform(self, X, sample_weight=None, average=True):
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    return self._transform_after_preprocessor(
      X, sample_weight=sample_weight, average=average)
  def _transform_after_preprocessor(self, X, sample_weight=None, average=True):
    if not average:
      fX = []
      for j in range(X.shape[1]): # feature index
        e = self.edges[j][1:]
        i_row = np.arange(X.shape[0])
        i_col = np.clip(np.ceil((X[:,j] - e[0]) / (e[1]-e[0])).astype(int), 0, self.n_bins-1)
        vals = np.ones(X.shape[0], dtype=float)
        if sample_weight is not None:
          vals *= sample_weight
        fX_j = csr_matrix(
          (vals, (i_row, i_col)),
          shape = (X.shape[0], self.n_bins),
        )
        fX.append(fX_j.toarray())
      fX = np.stack(fX).swapaxes(0, 1).reshape((X.shape[0], -1))
      if self.unit_scale:
        fX = fX / X.shape[1]
      return fX
    else: # a concatenation of numpy histograms is faster to compute
      histograms = []
      for j in range(X.shape[1]):  # feature index
        e = np.copy(self.edges[j])
        e[0] = -np.inf # always use exactly self.n_bins and never omit any items
        e[-1] = np.inf
        hist, _ = np.histogram(X[:, j], bins=e, weights=sample_weight)
        if self.unit_scale:
          hist = hist / X.shape[1]
        if sample_weight is not None:
          hist = hist / sample_weight.sum()
        else:
          hist = hist / X.shape[0]
        histograms.append(hist)
      return np.concatenate(histograms) # = q

@dataclass
class EnergyKernelRepresentation(AbstractRepresentation):
  """A kernel-based data representation, as it is used in `KMM`, that uses the `energy` kernel:

      k(x_1, x_2) = ||x_1|| + ||x_2|| - ||x_1 - x_2||

  Note:
      The methods of this representation do not support setting `average=False`.

  Args:
      preprocessor (optional): Another `AbstractRepresentation` that is called before this representation. Defaults to `None`.
  """
  preprocessor: Optional[AbstractRepresentation] = None
  def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
    if not average:
      raise ValueError("EnergyKernelRepresentation does not support average=False")
    if sample_weight is not None:
      raise ValueError("EnergyKernelRepresentation does not support sample_weight != None")

    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, sample_weight=sample_weight, average=False, n_classes=n_classes)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      check_y(y, n_classes)
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
  def transform(self, X, sample_weight=None, average=True):
    if not average:
      raise ValueError("EnergyKernelRepresentation does not support average=False")
    if sample_weight is not None:
      raise ValueError("EnergyKernelRepresentation does not support sample_weight != None")
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    norm = np.linalg.norm(X, axis=1).mean()
    return self._transform_after_preprocessor(X, norm)
  def _transform_after_preprocessor(self, X, norm, sample_weight=None):
    n_classes = len(self.p_trn)
    dists = np.zeros(n_classes) # = ||x_1 - x_2|| for all x_2 = X_trn[y_trn == i]
    for c in range(n_classes):
      if np.sum(self.y_trn==c) > 0:
        dists[c] = cdist(X, self.X_trn[self.y_trn==c], metric="euclidean").mean()
    return norm + self.norms - dists # = ||x_1|| + ||x_2|| - ||x_1 - x_2|| for all x_2

@dataclass
class GaussianKernelRepresentation(AbstractRepresentation):
  """A kernel-based data representation, as it is used in `KMM`, that uses the `gaussian` kernel:

      k(x, y) = exp(-||x - y||^2 / (2σ^2))

  Args:
      sigma (optional): A smoothing parameter of the kernel function. Defaults to `1`.
      preprocessor (optional): Another `AbstractRepresentation` that is called before this representation. Defaults to `None`.
  """
  sigma: float = 1.
  preprocessor: Optional[AbstractRepresentation] = None
  def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
    if not average:
      raise ValueError("GaussianKernelRepresentation does not support average=False")
    if sample_weight is not None:
      raise ValueError("GaussianKernelRepresentation does not support sample_weight != None")
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, sample_weight=sample_weight, average=False, n_classes=n_classes)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      check_y(y, n_classes)
      self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    self.X_trn = X
    self.y_trn = y
    M = np.zeros((n_classes, n_classes))
    for c in range(n_classes):
      M[:,c] = self._transform_after_preprocessor(X[y==c])
    return M
  def transform(self, X, sample_weight=None, average=True):
    if not average:
      raise ValueError("GaussianKernelRepresentation does not support average=False")
    if sample_weight is not None:
      raise ValueError("GaussianKernelRepresentation does not support sample_weight != None")
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    return self._transform_after_preprocessor(X)
  def _transform_after_preprocessor(self, X):
    n_classes = len(self.p_trn)
    res = np.zeros(n_classes)
    for i in range(n_classes):
      norm_fac = X.shape[0] * self.X_trn[self.y_trn==i].shape[0]
      sq_dists = cdist(X, self.X_trn[self.y_trn == i], metric="euclidean")**2
      res[i] = np.exp(-sq_dists / 2*self.sigma**2).sum() / norm_fac # <= old version
    return res

@dataclass
class KernelRepresentation(AbstractRepresentation):
  """A general kernel-based data representation, as it is used in `KMM`. If you intend to use a Gaussian kernel or energy kernel, prefer their dedicated and more efficient implementations over this class.

  Note:
      The methods of this representation do not support setting `average=False`.

  Args:
      kernel: A callable that will be used as the kernel. Must follow the signature `(X[y==i], X[y==j]) -> scalar`.
  """
  kernel: Callable
  def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
    if not average:
      raise ValueError("KernelRepresentation does not support average=False")
    if sample_weight is not None:
      raise ValueError("KernelRepresentation does not support sample_weight != None")
    check_y(y, n_classes)
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
  def transform(self, X, sample_weight=None, average=True):
    if not average:
      raise ValueError("KernelRepresentation does not support average=False")
    if sample_weight is not None:
      raise ValueError("KernelRepresentation does not support sample_weight != None")
    n_classes = len(self.p_trn)
    q = np.zeros(n_classes)
    for c in range(n_classes):
      if np.sum(self.y_trn==c) > 0:
        q[c] = self.kernel(self.X_trn[self.y_trn==c], X)
    return q

# kernel function for the LaplacianKernelRepresentation
def _laplacianKernel(X, Y, sigma):
    nx = X.shape[0]
    ny = Y.shape[0]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    Y = Y.reshape((1, Y.shape[0], Y.shape[1]))
    D_lk = np.abs(((X - Y))).sum(-1) 
    K_ij = np.exp((-sigma * D_lk)).sum(0).sum(0) / (nx * ny)
    return K_ij

@dataclass
class LaplacianKernelRepresentation(KernelRepresentation):
  """A kernel-based data representation, as it is used in `KMM`, that uses the `laplacian` kernel.

  Args:
      sigma (optional): A smoothing parameter of the kernel function. Defaults to `1`.
  """
  def __init__(self, sigma=1.):
    KernelRepresentation.__init__(self, LaplacianKernelRepresentation.create_kernel(sigma))
  @staticmethod
  def create_kernel(sigma):
    return partial(_laplacianKernel, sigma=sigma)

@dataclass
class GaussianRFFKernelRepresentation(AbstractRepresentation):
  """An efficient approximation of the `GaussianKernelRepresentation`, as it is used in `KMM`, using random Fourier features.

  Args:
      sigma (optional): A smoothing parameter of the kernel function. Defaults to `1`.
      n_rff (optional): The number of random Fourier features. Defaults to `1000`.
      preprocessor (optional): Another `AbstractRepresentation` that is called before this representation. Defaults to `None`.
      seed (optional): Controls the randomness of the random Fourier features. Defaults to `None`.
  """
  sigma: float = 1.
  n_rff: int = 1000
  preprocessor: Optional[AbstractRepresentation] = None
  seed: Optional[int] = None
  def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
    if not average:
      raise ValueError("GaussianRFFKernelRepresentation does not support average=False")
    if sample_weight is not None:
      raise ValueError("GaussianRFFKernelRepresentation does not support sample_weight != None")
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, sample_weight=sample_weight, average=False, n_classes=n_classes)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      check_y(y, n_classes)
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
  def transform(self, X, sample_weight=None, average=True):
    if not average:
      raise ValueError("GaussianRFFKernelRepresentation does not support average=False")
    if sample_weight is not None:
      raise ValueError("GaussianRFFKernelRepresentation does not support sample_weight != None")
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    fX = self._transform_after_preprocessor(X) @ self.mu
    return fX
  def _transform_after_preprocessor(self, X):
    Xw = X @ self.w.T
    C = np.concatenate((np.cos(Xw), np.sin(Xw)), axis=1)
    return np.sqrt(2 / self.n_rff) * np.mean(C, axis=0)

@dataclass
class OriginalRepresentation(AbstractRepresentation):
  """A dummy representation that simply returns the data as it is."""
  def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
    check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    if not average:
      return X, y
    if sample_weight is not None:
      M = np.array([ np.average(X[y==c], axis=0, weights=sample_weight[y==c]) for c in range(len(self.p_trn)) ]).T # = M
    else:
      M = np.array([ np.average(X[y==c], axis=0) for c in range(len(self.p_trn)) ]).T # = M
    return M
  def transform(self, X, sample_weight=None, average=True):
    n_classes = len(self.p_trn)
    if average:
      return np.average(X, axis=0, weights=sample_weight) # = q
    return X
