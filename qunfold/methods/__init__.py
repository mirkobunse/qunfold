import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from scipy import optimize

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

def minimize(
    fun,
    n_classes,
    solver = "trust-ncg",
    solver_options = {"gtol": 1e-8, "maxiter": 1000},
    seed = None,
  ):
  """Numerically minimize a function to predict the most likely class prevalences.

  This implementation makes use of a soft-max "trick" by Bunse (2022) and uses the auto-differentiation of JAX for second-order optimization.

  Args:
      fun: The function to minimize. Has to be implemented in JAX and has to have the signature `p -> loss`.
      n_classes: The number of classes.
      solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to `"trust-ncg"`.
      solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `{"gtol": 1e-8, "maxiter": 1000}`.
      seed (optional): A seed for random number generation. Defaults to `None`.

  Returns:
      A solution vector `p`.
  """
  fun_l = lambda l: fun(_jnp_softmax(l))
  jac = jax.grad(fun_l) # Jacobian
  hess = jax.jacfwd(jac) # Hessian through forward-mode AD
  x0 = _rand_x0(np.random.RandomState(seed), n_classes) # random starting point
  state = _CallbackState(x0)
  try:
    opt = optimize.minimize(
      fun_l,
      x0,
      jac = _check_derivative(jac, "jac"), # safe-guard derivatives
      hess = _check_derivative(hess, "hess"),
      method = solver,
      options = solver_options,
      callback = state.callback(),
    )
  except (DerivativeError, ValueError):
    traceback.print_exc()
    opt = state.get_state()
  return Result(_np_softmax(opt.x), opt.nit, opt.message)

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

def check_y(y, n_classes=None):
  """Emit a warning if the given labels are not sane."""
  if n_classes is not None:
    if n_classes != np.max(y)+1:
      warnings.warn(f"Classes are missing: n_classes != np.max(y)+1 = {np.max(y)+1}")

# helper function for our softmax "trick" with l[0]=0
def _jnp_softmax(l):
  exp_l = jnp.exp(l)
  return jnp.concatenate((jnp.ones(1), exp_l)) / (1. + exp_l.sum())
def _np_softmax(l): # like above but in numpy instead of JAX
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
    return lambda xk, *args: self._callback(xk)
