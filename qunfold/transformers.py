import numpy as np
from abc import ABC, abstractmethod

# helper function for crisp transformations
def _onehot_encoding(y, n_classes):
  return np.eye(n_classes)[y] # https://stackoverflow.com/a/42874726/20580159

class AbstractTransformer(ABC):
  """Abstract base class for transformers."""
  @abstractmethod
  def fit_transform(self, X, y):
    """Fit this transformer, set self.n_classes, and return a transformation (f(X), y)."""
    pass
  @abstractmethod
  def transform(self, X):
    """Transform X into f(X)."""
    pass

class ClassTransformer(AbstractTransformer):
  """This transformer yields the classification-based feature transformation used in ACC, PACC, CC, PCC, and SLD."""
  def __init__(self, classifier, is_probabilistic=False, fit_classifier=True):
    self.classifier = classifier
    self.is_probabilistic = is_probabilistic
    self.fit_classifier = fit_classifier
  def fit_transform(self, X, y):
    if not hasattr(self.classifier, "oob_score") or not self.classifier.oob_score:
      raise ValueError("Only bagging classifiers with oob_score=True are supported")
    if self.fit_classifier:
      if y.min() not in [0, 1]:
        raise ValueError("y.min() ∉ [0, 1]")
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
