import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

# helper function for crisp transformations
def _onehot_encoding(y, n_classes):
  return np.eye(n_classes)[y] # https://stackoverflow.com/a/42874726/20580159

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

class AbstractTransformer(ABC):
  """Abstract base class for transformers."""
  @abstractmethod
  def fit_transform(self, X, y, average=True):
    """This abstract method has to fit the transformer and to return the transformation of the input data.

    Note:
        Implementations of this abstract method must set the property `self.p_trn = class_prevalences(y)`.

    Args:
        X: The feature matrix to which this transformer will be fitted.
        y: The labels to which this transformer will be fitted.
        average (optional): Whether to return a transfer matrix `M` or a transformation `(f(X), y)`. Defaults to `True`.

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
  @property
  def n_classes(self):
    return len(self.p_trn)

class ClassTransformer(AbstractTransformer):
  """A classification-based feature transformation.

  This transformation can either be probabilistic (using the posterior predictions of a classifier) or crisp (using the class predictions of a classifier). It is used in ACC, PACC, CC, PCC, and SLD.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      is_probabilistic (optional): Whether probabilistic or crisp predictions of the `classifier` are used to transform the data. Defaults to `False`.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
  """
  def __init__(self, classifier, is_probabilistic=False, fit_classifier=True):
    self.classifier = classifier
    self.is_probabilistic = is_probabilistic
    self.fit_classifier = fit_classifier
  def fit_transform(self, X, y, average=True):
    if y.min() not in [0, 1]:
      raise ValueError("y.min() ∉ [0, 1]")
    if not hasattr(self.classifier, "oob_score") or not self.classifier.oob_score:
      raise ValueError(
        "The ClassTransformer either requires a bagging classifier with oob_score=True",
        "or an instance of qunfold.sklearn.CVClassifier"
      )
    if self.fit_classifier:
      self.classifier.fit(X, y)
    fX = self.classifier.oob_decision_function_
    is_finite = np.all(np.isfinite(fX), axis=1)
    fX = fX[is_finite,:]
    y = y[is_finite] - y.min() # map to zero-based labels
    self.p_trn = class_prevalences(y) # also sets self.n_classes correctly
    if not self.is_probabilistic:
      fX = _onehot_encoding(np.argmax(fX, axis=1), self.n_classes)
    if average:
      M = np.zeros((fX.shape[1], self.n_classes))
      for c in range(self.n_classes):
        M[:,c] = fX[y==c].mean(axis=0)
      return M
    return fX, y
  def transform(self, X, average=True):
    fX = self.classifier.predict_proba(X)
    if not self.is_probabilistic:
      fX = _onehot_encoding(np.argmax(fX, axis=1), self.n_classes)
    if average:
        fX = fX.mean(axis=0)
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
  def fit_transform(self, X, y, average=True):
    if y.min() not in [0, 1]:
      raise ValueError("y.min() ∉ [0, 1]")
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, average=False)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      y -= y.min() # map to zero-based labels
      self.p_trn = class_prevalences(y) # also sets self.n_classes correctly
    self.X_trn = X
    self.y_trn = y
    if average:
      M = np.zeros((self.n_classes, self.n_classes))
      for c in range(self.n_classes):
        M[:,c] = self._transform_after_preprocessor(X[y==c])
      return M
    else:
      return self._transform_after_preprocessor(X, average=False), y
  def transform(self, X, average=True):
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X, average=False)
    return self._transform_after_preprocessor(X, average=average)
  def _transform_after_preprocessor(self, X, average=True):
    fX = np.zeros((X.shape[0], self.n_classes))
    for i in range(self.n_classes): # class index
      fX[:, i] = cdist(X, self.X_trn[self.y_trn == i], metric=self.metric).mean(axis=1)
    if average:
      fX = fX.mean(axis=0)
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
  def fit_transform(self, X, y, average=True):
    if y.min() not in [0, 1]:
      raise ValueError("y.min() ∉ [0, 1]")
    if self.preprocessor is not None:
      X, y = self.preprocessor.fit_transform(X, y, average=False)
      self.p_trn = self.preprocessor.p_trn # copy from preprocessor
    else:
      y -= y.min() # map to zero-based labels
      self.p_trn = class_prevalences(y) # also sets self.n_classes correctly
    self.edges = []
    for x in X.T: # iterate over columns = features
      e = np.histogram_bin_edges(x, bins=self.n_bins)
      self.edges.append(e)
    if average:
      M = np.zeros((X.shape[1] * self.n_bins, self.n_classes))
      for c in range(self.n_classes):
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
      return np.concatenate(histograms)
    
# Kernelfunction for EnergyKernelTransformer
def _energyKernelProduct(X, Y, sigma=1):  # sigma not needed but added for consistent kernel signature
    nx = X.shape[0]
    ny = Y.shape[0]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    Y = Y.reshape((1, Y.shape[0], Y.shape[1]))
    norm_x = np.sqrt((X**2).sum(-1)).sum(0) / nx
    norm_y = np.sqrt((Y**2).sum(-1)).sum(1) / ny
    Dlk = np.sqrt(((X - Y)**2).sum(-1))
    return np.squeeze(norm_x + norm_y) - Dlk.sum(0).sum(0) / (nx * ny)
# Kernelfunction for GaussianKernelTransformer
def _gaussianKernelProduct(X, Y, sigma=1):
    nx = X.shape[0]
    ny = Y.shape[0]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    Y = Y.reshape((1, Y.shape[0], Y.shape[1]))
    D_lk = ((X - Y)**2).sum(-1)
    K_ij = np.exp((-D_lk / (2 * sigma**2))).sum(0).sum(0) / (nx * ny)
    return K_ij
# Kernelfunction for GaussianKernelTransformer
def _laplacianKernelProduct(X, Y, sigma=1):
    nx = X.shape[0]
    ny = Y.shape[0]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    Y = Y.reshape((1, Y.shape[0], Y.shape[1]))
    D_lk = np.abs(((X - Y))).sum(-1) 
    K_ij = np.exp((-sigma * D_lk)).sum(0).sum(0) / (nx * ny)
    return K_ij

class KernelTransformer:
  """A kernel-based feature transformation, as it is used in `KMM`.
  
  Args:
    kernel: a callable that will be used as the kernel. Must follow the signature ``k(X, Y, sigma)``.
    sigma: a necessary parameter to calculate `gaussian` and `laplacian` kernel. Defaults to `1`
  """
  def __init__(self, kernel, sigma=1) -> None:
    self.kernel = kernel
    self.sigma = sigma
  def fit_transform(self, X, y):
    self.n_classes = len(np.unique(y))
    self.X_trn = X
    self.y_trn = y
    self.p_trn = np.zeros((self.n_classes))
    for c in range(self.n_classes):
      self.p_trn[c] = (y==c).sum() / y.shape[0]
    M = np.zeros((self.n_classes, self.n_classes))
    for i in range(self.n_classes):
      for j in range(i, self.n_classes):
        M[i, j] = self.kernel(X[y==i], X[y==j], sigma=self.sigma)
        if i != j:
            M[j, i] = M[i, j]
    return M
  def transform(self, X):
    q = np.zeros((self.n_classes))
    for i in range(self.n_classes):
      q[i] = self.kernel(self.X_trn[self.y_trn==i], X, sigma=self.sigma)
    return q

class EnergyKernelTransformer(KernelTransformer):
  """A kernel-based feature transformation as it is used in `KMM`, that uses the `energy` kernel\n
  k(x, y) = ||x|| + ||y|| - ||x - y||.

  Args:
    sigma: unused parameter that is kept to keep consistent signature across supported kernels. Defaults to `1`.
  """
  def __init__(self, sigma=1) -> None:
    KernelTransformer.__init__(
      self,
      kernel=_energyKernelProduct
    )
  
class GaussianKernelTransformer(KernelTransformer):
  """A kernel-based feature transformation as it is used in `KMM`, that uses the `gaussian` kernel.

  Args:
    sigma: necessary parameter to calculate the `gaussian` kernel.
  """
  def __init__(self, sigma=1) -> None:
    KernelTransformer.__init__(
      self,
      kernel=_gaussianKernelProduct,
      sigma=sigma
    )
  
class LaplacianKernelTransformer(KernelTransformer):
  """A kernel-based feature transformation as it is used in `KMM`, that uses the `laplacian` kernel.

  Args:
    sigma: necessary parameter to calculate the `laplacian` kernel.
  """
  def __init__(self, sigma=1) -> None:
    KernelTransformer.__init__(
      self,
      kernel=_laplacianKernelProduct,
      sigma=sigma
    )