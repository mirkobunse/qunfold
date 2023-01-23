import numpy as np
import traceback
from scipy import optimize
from . import (losses, transformers)

# helper function for our softmax "trick" with l[0]=0
def _np_softmax(l):
  exp_l = np.exp(l)
  return np.concatenate((np.ones(1), exp_l)) / (1. + exp_l.sum())

# helper function for random starting points
def _rand_x0(rng, n_classes):
  return rng.rand(n_classes-1) * 2 - 1

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

# helpers for maintaining the last result in case of an error
class DerivativeError(Exception):
  def __init__(self, name, result):
    super().__init__(f"infs and NaNs in {name}: {result}")
def _check_derivative(jac_or_hess, name):
  return lambda x: _check_derivative_at_x(jac_or_hess, name, x)
def _check_derivative_at_x(jac_or_hess, name, x):
  result = jac_or_hess(x)
  if not np.all(np.isfinite(result)):
    raise DerivativeError(name, result)
  return result

class _CallbackState():
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

class GenericMethod:
  """A generic quantification / unfolding method."""
  def __init__(self, loss, transformer, solver="trust-ncg", seed=None):
    self.loss = loss
    self.transformer = transformer
    self.solver = solver
    self.seed = seed
  def fit(self, X, y):
    fX, fy = self.transformer.fit_transform(X, y) # f(x) for x ∈ X
    M = np.zeros((fX.shape[1], self.transformer.n_classes)) # (n_features, n_classes)
    for c in range(self.transformer.n_classes):
      M[:,c] = fX[fy==c,:].sum(axis=0) # one histogram of f(X) per class
    self.M = M / M.sum(axis=0, keepdims=True)
    self.p_trn = M.sum(axis=0) / M.sum()
    return self
  def predict(self, X):
    q = self.transformer.transform(X).mean(axis=0)
    return self.solve(q, self.M)
  def solve(self, q, M): # TODO add arguments p_trn and N=X.shape[0]
    loss_dict = losses.instantiate_loss(self.loss, q, M)
    rng = np.random.RandomState(self.seed)
    x0 = _rand_x0(rng, M.shape[1]) # random starting point
    state = _CallbackState(x0)
    try:
      opt = optimize.minimize(
        loss_dict["fun"], # JAX function l -> loss
        x0,
        jac = _check_derivative(loss_dict["jac"], "jac"),
        hess = _check_derivative(loss_dict["hess"], "hess"),
        method = self.solver,
        callback = state.callback()
      )
    except (DerivativeError, ValueError):
      traceback.print_exc()
      opt = state.get_state()
    return Result(_np_softmax(opt.x), opt.nit, opt.message)

class ACC(GenericMethod):
  """Adjusted Classify & Count."""
  def __init__(self, classifier, fit_classifier=True, **kwargs):
    GenericMethod.__init__(
      self,
      losses.LeastSquaresLoss(),
      transformers.ClassTransformer(
        classifier,
        fit_classifier = fit_classifier
      ),
      **kwargs
    )

class PACC(GenericMethod):
  """Probabilistic Adjusted Classify & Count."""
  def __init__(self, classifier, fit_classifier=True, **kwargs):
    GenericMethod.__init__(
      self,
      losses.LeastSquaresLoss(),
      transformers.ClassTransformer(
        classifier,
        fit_classifier = fit_classifier,
        is_probabilistic = True
      ),
      **kwargs
    )
