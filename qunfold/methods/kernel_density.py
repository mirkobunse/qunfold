import jax
import jax.numpy as jnp
import numpy as np
import traceback
from scipy.optimize import minimize
from scipy.stats import scoreatpercentile
from sklearn.neighbors import KernelDensity
from qunfold.losses import KDEyMLLoss, instantiate_loss
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

class KDEBase:
  def __init__(self, classifier, bandwidth, random_state=None, solver='SLSQP') -> None:
    self.classifier = classifier
    self.bandwidth = bandwidth
    self.random_state = random_state
    self.solver = solver

  def fit(self, X, y, n_classes=None):
    check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    self.n_classes = len(self.p_trn) # not None anymore
    self.preprocessor = ClassTransformer(classifier=self.classifier, is_probabilistic=True)
    fX, _ = self.preprocessor.fit_transform(X, y, average=False)
    #self.classifier.fit(X, y)
    #fX = getattr(self.classifier, 'decision_function')(X)
    if isinstance(self.bandwidth, list) or isinstance(self.bandwidth, np.ndarray):
      assert len(self.bandwidth) == self.n_classes, (
        f"bandwidth must either be a single scalar or a sequence of length n_classes.\n"
        f"Received {len(self.bandwidth)} values for bandwidth, but dataset has {n_classes} classes."
      )
      self.mixture_components = [
          KernelDensity(bandwidth=self.bandwidth[c]).fit(fX[y==c])
          for c in range(self.n_classes)
        ]
    else:
      self.mixture_components = [
          KernelDensity(bandwidth=self.bandwidth).fit(fX[y==c])
          for c in range(self.n_classes)
        ]
    
    return self
  
  def predict(self, X):
    fX = self.preprocessor.transform(X, average=False)
    #fX = getattr(self.classifier, 'decision_function')(X)
    return self.solve(fX)
  
  def solve(self, X):
    pass

class KDEyMLQP(KDEBase):

  def __init__(self, classifier, bandwidth, random_state=None, solver='SLSQP') -> None:
    KDEBase.__init__(
      self,
      classifier = classifier,
      bandwidth = bandwidth,
      random_state = random_state,
      solver = solver,
    )

  def solve(self, X):
    np.random.RandomState(self.random_state)
    epsilon = 1e-10
    n_classes = len(self.mixture_components)
    test_densities = [np.exp(mc.score_samples(X)) for mc in self.mixture_components]

    def neg_loglikelihood(prevs):
      test_mixture_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip(prevs, test_densities))
      test_loglikelihood = np.log(test_mixture_likelihood + epsilon)
      return -np.sum(test_loglikelihood)
    
    x0 = np.full(fill_value=1 / n_classes, shape=(n_classes,))
    bounds = tuple((0, 1) for _ in range(n_classes))
    constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    opt = minimize(
      neg_loglikelihood,
      x0=x0,
      method=self.solver,
      bounds=bounds,
      constraints=constraints
    )
    return Result(opt.x, opt.nit, opt.message)
  
# assume that bandwidth is either float/int or 'scott' or 'silverman'
# cant handle list/array, since bandwidth is used in solve
class KDEyHDQP(KDEBase):
  def __init__(self, classifier, bandwidth, random_state=None, solver='SLSQP', montecarlo_trials=10_000) -> None:
    super().__init__(
      self,
      classifier, 
      bandwidth, 
      random_state, 
      solver
    )
    self.montecarlo_trials = montecarlo_trials

  def fit(self, X, y, n_classes=None):
    super().fit(self, X, y, n_classes)
    N = self.montecarlo_trials
    rs = self.random_state
    self.reference_samples = np.vstack([
      mc.sample(N//self.n_classes, random_state=rs) 
      for mc in self.mixture_components
    ])
    self.reference_classwise_densities = np.asarray([
      np.exp(mc.score_samples(self.reference_samples)) 
      for mc in self.mixture_components
    ])
    self.reference_density = np.mean(self.reference_classwise_densities, axis=0)
    return self

  def solve(self, X):
    if self.bandwidth == 'scott':
      test_kde = KernelDensity(bandwidth=_bw_scott(X)).fit(X)
    elif self.bandwidth == 'silverman':
      test_kde = KernelDensity(bandwidth=_bw_silverman(X)).fit(X)
    else:
      test_kde = KernelDensity(bandwidth=self.bandwidth).fit(X)
      
    test_densities = np.exp(test_kde.score_samples(self.reference_samples))

    def f_squared_hellinger(u):
      return (np.sqrt(u) - 1)**2
      
    epsilon = 1e-10
    qs = test_densities + epsilon
    rs = self.reference_density + epsilon
    iw = qs/rs
    p_class = self.reference_classwise_densities + epsilon
    fracs = p_class/qs

    def divergence(prev):
      ps_div_qs = prev @ fracs
      return np.mean(f_squared_hellinger(ps_div_qs) * iw)
      
    x0 = np.full(fill_value=1 / self.n_classes, shape=(self.n_classes,))
    bounds = tuple((0, 1) for _ in range(self.n_classes))
    constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})

    opt = minimize(
      divergence,
      x0=x0,
      method=self.solver,
      bounds=bounds,
      constraints=constraints
    )
      
    return Result(opt.x, opt.nit, opt.message)
    