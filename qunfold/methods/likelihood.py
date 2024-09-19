import jax.numpy as jnp
from . import AbstractMethod, check_y, class_prevalences, minimize

class LikelihoodMaximizer(AbstractMethod):
  """The maximum likelihood method, as studied by Alexandari et al. (2020).

  This method is proven to be asymptotically equivalent to the `ExpectationMaximizer` by Saerens et al. (2002).

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to `"trust-ncg"`.
      solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `{"gtol": 1e-8, "maxiter": 1000}`.
      tau_0 (optional): The regularization strength for penalizing deviations from uniform predictions. Defaults to `0`.
      tau_1 (optional): The regularization strength for penalizing deviations from non-ordinal predictions. Defaults to `0`.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      seed (optional): A random number generator seed from which a numpy RandomState is created. Defaults to `None`.
  """
  def __init__(
      self,
      classifier,
      solver = "trust-ncg",
      solver_options = {"gtol": 1e-8, "maxiter": 1000}, # , "disp": True
      tau_0 = 0,
      tau_1 = 0,
      epsilon = 0,
      fit_classifier = True,
      seed = None,
      ):
    self.classifier = classifier
    self.solver = solver
    self.solver_options = solver_options
    self.tau_0 = tau_0
    self.tau_1 = tau_1
    self.epsilon = epsilon
    self.fit_classifier = fit_classifier
    self.seed = seed
  def fit(self, X, y, n_classes=None):
    check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    if self.fit_classifier:
      self.classifier.fit(X, y)
    return self
  def predict(self, X):
    pXY = classifier.predict_proba(X) / self.p_trn # proportional to P(X|Y)
    pXY = pXY / pXY.sum(axis=1, keepdims=True) # normalize to P(X|Y)

    # TODO 1) filter out all rows from pXY that contain zeros or ones, or values close to zero or one up to some self.epsilon. Goal: to reduce thrown errors / warnings and to replace the corresponding estimates with proper ones.
    #epsilon_filtered_rows = pXY[np.any(pXY <= self.epsilon, axis=1), :]
    #pXY = pXY[np.all(pXY > self.epsilon, axis=1),:]

    # TODO 2) "side-chain" those rows that have contained values close to one, by setting up a classify-and-count estimate that is later added to opt.x. Appropriately weight both the CC estimate and opt.x by the fraction of rows that has lead to each of these estimates. Goal: to further improve the estimation (see also the todo 3).
    
    # TODO 3) consider self.epsilon as a hyper-parameter, assuming that all fairly confident predictions are probably correct, not only the extemely confident exceptions.

    # minimize the negative log-likelihood loss
    def loss(p):
      xi_0 = jnp.sum((p[1:] - p[:-1])**2) / 2 # deviation from a uniform prediction
      xi_1 = jnp.sum((-p[:-2] + 2 * p[1:-1] - p[2:])**2) / 2 # deviation from non-ordinal
      return -jnp.log(pXY @ p).mean() + self.tau_0 * xi_0 + self.tau_1 * xi_1
    return minimize(
      loss,
      len(self.p_trn),
      np.random.RandomState(self.seed), # = rng
      self.solver,
      self.solver_options
    )

class ExpectationMaximizer(AbstractMethod):
  """The expectation maximization-based method  by Saerens et al. (2002).

  This method is proven to be asymptotically equivalent to the `LikelihoodMaximizer` by Alexandari et al. (2020).

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      max_iter (optional): The maximum number of iterations. Defaults to `100`.
      tol (optional): The convergence tolerance for the L2 norm between iterations. Defaults to `1e-8`.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
  """
  def __init__(
      self,
      classifier,
      max_iter = 100,
      tol = 1e-8,
      fit_classifier = True,
      ):
    self.classifier = classifier
    self.max_iter = max_iter
    self.tol = tol
    self.fit_classifier = fit_classifier
  def fit(self, X, y, n_classes=None):
    check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    if self.fit_classifier:
      self.classifier.fit(X, y)
    return self
  def predict(self, X):
    pXY = classifier.predict_proba(X) / self.p_trn # proportional to P(X|Y)
    pXY = pXY / pXY.sum(axis=1, keepdims=True) # normalize to P(X|Y)
    raise NotImplementedError("TODO")
