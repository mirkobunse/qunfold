import numpy as np
from abc import ABC, abstractmethod

# helper function for crisp transformations
def _onehot_encoding(y, n_classes):
  return np.eye(n_classes)[y] # https://stackoverflow.com/a/42874726/20580159

class AbstractTransformer(ABC):
  """Abstract base class for transformers."""
  @abstractmethod
  def fit_transform(self, X, y):
    """This abstract method has to the transformer to data and return a transformation `(f(X), y)`.

    Note:
        Implementations of this abstract method must set the property `self.n_classes`.

    Args:
        X: The feature matrix to which this transformer will be fitted.
        y: The labels to which this transformer will be fitted.

    Returns:
        A transformation `(f(X), y)`.
    """
    pass
  @abstractmethod
  def transform(self, X):
    """This abstract method has to transform `X` into `f(X)`.

    Args:
        X: The feature matrix that will be transformed.

    Returns:
        A transformation `f(X)` of this feature matrix.
    """
    pass

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
  def fit_transform(self, X, y):
    if y.min() not in [0, 1]:
      raise ValueError("y.min() ∉ [0, 1]")
    if not hasattr(self.classifier, "oob_score") or not self.classifier.oob_score:
      raise ValueError(
        "The ClassTransformer either requires a bagging classifier with oob_score=True",
        "or an instance of qunfold.sklearn.CVClassifier"
      )
    if self.fit_classifier:
      self.classifier.fit(X, y)
    self.n_classes = len(self.classifier.classes_)
    fX = self.classifier.oob_decision_function_
    is_finite = np.all(np.isfinite(fX), axis=1)
    fX = fX[is_finite,:]
    y = y[is_finite] - y.min() # map to zero-based labels
    if not self.is_probabilistic:
      fX = _onehot_encoding(np.argmax(fX, axis=1), self.n_classes)
    return fX, y
  def transform(self, X):
    fX = self.classifier.predict_proba(X)
    if self.is_probabilistic:
      return fX
    else:
      return _onehot_encoding(np.argmax(fX, axis=1), self.n_classes)

class HistogramTransformer(AbstractTransformer):
  """A histogram-based feature transformation, as it is used in `HDx` and `HDy`.

  Args:
      n_bins: The number of bins in each feature.
      preprocessor (optional): Another `AbstractTransformer` that is called before this transformer. Defaults to `None`.
      unit_scale (optional): Whether or not to scale each output to a sum of one. Defaults to `False`, indicating that the sum of each output is the number of features.
  """
  def __init__(self, n_bins, preprocessor=None, unit_scale=False):
    self.n_bins = n_bins
    self.preprocessor = preprocessor
    self.unit_scale = unit_scale
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
      e[-1] = np.inf
      self.edges.append(e)
    return self._transform_after_preprocessor(X), y
  def transform(self, X):
    if self.preprocessor is not None:
      X = self.preprocessor.transform(X)
    return self._transform_after_preprocessor(X)
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
