import jax
import jax.numpy as jnp
import numpy as np
import traceback
from scipy.optimize import minimize
from scipy.stats import scoreatpercentile
from sklearn.neighbors import KernelDensity
from . import (
  AbstractMethod,
  rand_x0,
  np_softmax,
  Result,
  DerivativeError,
  check_derivative,
  MinimizeCallbackState
)
from ..transformers import (check_y, class_prevalences, ClassTransformer)

def _bw_scott(X):
  sigma = np.std(X, ddof=1)
  return 3.49 * sigma * X.shape[0]**(-0.333)

def _bw_silverman(X):
  norm_iqr = (scoreatpercentile(X, 75) - scoreatpercentile(X, 25)) / 1.349
  sigma = np.std(X, ddof=1)
  A = np.minimum(sigma, norm_iqr) if norm_iqr > 0 else sigma
  return 0.9 * A * X.shape[0]**(-0.2)

class KDEyML(AbstractMethod):
  """The Maximum-Likelihood solution of the kernel-based KDE method by González-Moreo et al. (2024).

  In this implementation, the method employs unconstrained second-order minimization. Valid probability estimates are ensured through a soft-max trick by Bunse (2022).

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      bandwith: A smoothing parameter for the kernel-function. Either a single bandwidth for all classes, or a sequenz of individual values for each class.
      solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to `"trust-ncg"`.
      solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `{"gtol": 1e-8, "maxiter": 1000}`.
      seed (optional): A random number generator seed from which a numpy RandomState is created. Defaults to `None`.
  """
  def __init__(self,
      classifier,
      bandwidth,
      solver = "trust-ncg",
      solver_options = {"gtol": 1e-8, "maxiter": 1000},
      seed = None,
      ):
    self.classifier = classifier
    self.bandwidth = bandwidth
    self.solver = solver
    self.solver_options = solver_options
    self.seed = seed
  def fit(self, X, y, n_classes=None):
    check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    if isinstance(self.bandwidth, float) or isinstance(self.bandwidth, int):
      self.bandwidth = [self.bandwidth] * n_classes
    elif isinstance(self.bandwidth, str):
      self.bandwidth = self.bandwidth.lower()
      assert self.bandwidth == 'scott' or self.bandwidth == 'silverman', (
        f"Valid bandwidth estimation methods are 'scott' and 'silverman', got {self.bandwidth}!"
      )
    else:
      assert len(self.bandwidth) == n_classes, (
        f"bandwidth must either be a single scalar or a sequence of length n_classes.\n"
        f"Received {len(self.bandwidth)} values for bandwidth, but dataset has {n_classes} classes."
      )
    self.preprocessor = ClassTransformer(
      self.classifier,
      is_probabilistic = True,
      fit_classifier = True
    )
    fX, _ = self.preprocessor.fit_transform(X, y, average=False)
    if isinstance(self.bandwidth, str) and self.bandwidth == 'silverman':
      self.mixture_components = [
      KernelDensity(bandwidth=_bw_silverman(fX[y==c])).fit(fX[y==c])
      for c in range(n_classes)
      ]
    elif isinstance(self.bandwidth, str) and self.bandwidth == 'scott':
      self.mixture_components = [
        KernelDensity(bandwidth=_bw_scott(fX[y==c])).fit(fX[y==c])
        for c in range(n_classes)
      ]
    else:
      self.mixture_components = [
        KernelDensity(bandwidth=self.bandwidth[c]).fit(fX[y==c])
        for c in range(n_classes)
      ]
    return self
  def predict(self, X):
    fX = self.preprocessor.transform(X, average=False)
    q = jnp.vstack( # log probabilities, shape (n_samples, n_classes)
      [ mc.score_samples(fX) for mc in self.mixture_components],
      dtype = jnp.float32
    ).T

    # # Eq. 18 in Moreo et al. (2024)
    # q = jnp.exp(jnp.vstack( # probabilities, shape (n_samples, n_classes)
    #   [ mc.score_samples(fX) for mc in self.mixture_components],
    #   dtype = jnp.float32
    # ).T)
    # def fun(l):
    #   exp_l = jnp.exp(l)
    #   p = jnp.concatenate((jnp.ones(1), exp_l)) / (1. + exp_l.sum())
    #   return -jnp.log(jnp.dot(q, p)).mean()

    # # a variant of Eq. 18 in Moreo et al. (2024), where q and l are logarithmic
    scaling = jnp.log(X.shape[0]) # scale to implement averaging inside the logarithm
    def fun(l):
      l = jnp.concatenate((jnp.zeros(1), l)) # l[0] = 0
      l = l - jax.scipy.special.logsumexp(l) # normalize
      return -jax.scipy.special.logsumexp(q + l - scaling, axis=1).sum()
    # def fun(l):
    #   exp_ql = jnp.exp(q + jnp.concatenate((jnp.zeros(1), l)))
    #   return -jnp.log(exp_ql.sum(axis=1)).mean()
    # def fun(l):
    #   exp_ql = jnp.exp(q + jnp.concatenate((jnp.zeros(1), l)))
    #   p = exp_ql / exp_ql.sum(axis=1, keepdims=True)
    #   return -p.mean()
    # anchor = jnp.ones(1) * jnp.log(1 / len(self.mixture_components))
    # def fun(l):
    #   l = l - jax.scipy.special.logsumexp(jnp.concatenate((anchor, l))) # normalize
    #   return -jax.scipy.special.logsumexp(
    #     q + jnp.concatenate((anchor, l)) - scaling, # l[0] = 0
    #     axis = 1
    #   ).sum()
    # fun = lambda l: -jax.scipy.special.logsumexp(
    #   q + jnp.concatenate((anchor, l)) - scaling, # l[0] = 0
    #   axis = 1
    # ).sum()
    jac = jax.grad(fun)
    hess = jax.jacfwd(jac) # forward-mode AD

    # optimize
    rng = np.random.RandomState(self.seed)
    x0 = rand_x0(rng, len(self.mixture_components)) # random starting point
    # x0 = anchor + rand_x0(rng, len(self.mixture_components)) # random starting point
    state = MinimizeCallbackState(x0)
    try:
      opt = minimize(
        fun,
        x0,
        jac = check_derivative(jac, "jac"),
        hess = check_derivative(hess, "hess"),
        method = self.solver,
        options = self.solver_options,
        callback = state.callback()
      )
    except (DerivativeError, ValueError):
      traceback.print_exc()
      opt = state.get_state()
    # exp_l = jnp.exp(jnp.concatenate((anchor, opt.x)))
    # return Result(np.array(exp_l / exp_l.sum()), opt.nit, opt.message)
    return Result(np_softmax(opt.x), opt.nit, opt.message)

