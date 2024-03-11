import numpy as np
from abc import ABC, abstractmethod

class AbstractMethod(ABC):
  """Abstract base class for quantification methods."""
  @abstractmethod
  def fit(self, X, y, n_classes=None):
    """Fit this quantifier to data.

    Args:
        X: The feature matrix to which this quantifier will be fitted.
        y: The labels to which this quantifier will be fitted.
        n_classes (optional): The number of expected classes. Defaults to `None`.

    Returns:
        This fitted quantifier itself.
    """
    pass
  @abstractmethod
  def predict(self, X):
    """Predict the class prevalences in a data set.

    Args:
        X: The feature matrix for which this quantifier will make a prediction.

    Returns:
        A numpy array of class prevalences.
    """
    pass

class Result(np.ndarray): # https://stackoverflow.com/a/67510022/20580159
  """A numpy array with additional properties nit and message."""
  def __new__(cls, input_array, nit, message):
    obj = np.asarray(input_array).view(cls)
    obj.nit = nit
    obj.message = message
    return obj
  def __array_finalize__(self, obj):
    if obj is None:
      return
    self.nit = getattr(obj, "nit", None)
    self.message = getattr(obj, "message", None)

# several utilities for maintaining the latest result in case of an error
class DerivativeError(Exception):
  def __init__(self, name, result):
    super().__init__(f"infs and NaNs in {name}: {result}")

def check_derivative(jac_or_hess, name):
  return lambda x: _check_derivative_at_x(jac_or_hess, name, x)
def _check_derivative_at_x(jac_or_hess, name, x):
  result = jac_or_hess(x)
  if not np.all(np.isfinite(result)):
    raise DerivativeError(name, result)
  return result

class MinimizeCallbackState():
  def __init__(self, x0):
    self._xk = x0
    self._nit = 0
  def get_state(self):
    return optimize.OptimizeResult(
      x = self._xk,
      success = False,
      message = "Intermediate result",
      nit = self._nit
    )
  def _callback(self, xk):
    self._xk = xk
    self._nit += 1
  def callback(self):
    return lambda xk: self._callback(xk)
