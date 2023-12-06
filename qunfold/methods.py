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
  """A generic quantification / unfolding method.

  This class represents any method that consists of a loss function, a feature transformation, and a regularization term. In this implementation, any regularized loss is minimized through unconstrained second-order minimization. Valid probability estimates are ensured through a soft-max trick by Bunse (2022).

  Args:
      loss: An instance from `qunfold.losses`.
      transformer: An instance from `qunfold.transformers`.
      solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to `"trust-ncg"`.
      solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `{"gtol": 1e-8, "maxiter": 1000}`.
      seed (optional): A random number generator seed from which a numpy RandomState is created. Defaults to `None`.

  Examples:
      Here, we create the ordinal variant of ACC (Bunse et al., 2023). This variant consists of the original feature transformation of ACC and of the original loss of ACC, the latter of which is regularized towards smooth solutions.

          >>> GenericMethod(
          >>>     TikhonovRegularized(LeastSquaresLoss(), 0.01),
          >>>     ClassTransformer(RandomForestClassifier(oob_score=True))
          >>> )
  """
  def __init__(self, loss, transformer,
      solver = "trust-ncg",
      solver_options = {"gtol": 1e-8, "maxiter": 1000},
      seed = None,
      ):
    self.loss = loss
    self.transformer = transformer
    self.solver = solver
    self.solver_options = solver_options
    self.seed = seed
  def fit(self, X, y):
    """Fit this quantifier to data.

    Args:
        X: The feature matrix to which this quantifier will be fitted.
        y: The labels to which this quantifier will be fitted.

    Returns:
        This fitted quantifier itself.
    """
    self.M = self.transformer.fit_transform(X, y)
    return self
  def predict(self, X):
    """Predict the class prevalences in a data set.

    Args:
        X: The feature matrix for which this quantifier will make a prediction.

    Returns:
        A numpy array of class prevalences.
    """
    q = self.transformer.transform(X)
    return self.solve(q, self.M, N=X.shape[0])
  def solve(self, q, M, N=None): # TODO add argument p_trn=self.p_trn
    """Solve the linear system of equations `q=M*p` for `p`.

    Args:
        q: A numpy array.
        M: A numpy matrix.
        N: The number of data items that `q` represents. For some losses, this argument is optional.

    Returns:
        The solution vector `p`.
    """
    loss_dict = losses.instantiate_loss(self.loss, q, M, N)
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
        options = self.solver_options,
        callback = state.callback()
      )
    except (DerivativeError, ValueError):
      traceback.print_exc()
      opt = state.get_state()
    return Result(_np_softmax(opt.x), opt.nit, opt.message)
  @property
  def p_trn(self):
    return self.transformer.p_trn

class ACC(GenericMethod):
  """Adjusted Classify & Count by Forman (2008).

  This subclass of `GenericMethod` is instantiated with a `LeastSquaresLoss` and a `ClassTransformer`.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      **kwargs: Keyword arguments accepted by `GenericMethod`.
  """
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
  """Probabilistic Adjusted Classify & Count by Bella et al. (2010).

  This subclass of `GenericMethod` is instantiated with a `LeastSquaresLoss` and a `ClassTransformer`.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      **kwargs: Keyword arguments accepted by `GenericMethod`.
  """
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

class RUN(GenericMethod):
  """Regularized Unfolding by Blobel (1985).

  This subclass of `GenericMethod` is instantiated with a `TikhonovRegularized(BlobelLoss)`.

  Args:
      transformer: An instance from `qunfold.transformers`.
      tau (optional): The regularization strength. Defaults to 0.
      **kwargs: Keyword arguments accepted by `GenericMethod`.
  """
  def __init__(self, transformer, *, tau=0., **kwargs):
    GenericMethod.__init__(
      self,
      losses.TikhonovRegularized(losses.BlobelLoss(), tau),
      transformer,
      **kwargs
    )

class EDx(GenericMethod):
  """The energy distance-based EDx method by Kawakubo et al. (2016).

  This subclass of `GenericMethod` is instantiated with an `EnergyLoss` and a `DistanceTransformer`.

  Args:
      metric (optional): The metric with which the distance between data items is measured. Can take any value that is accepted by `scipy.spatial.distance.cdist`. Defaults to `"euclidean"`.
      **kwargs: Keyword arguments accepted by `GenericMethod`.
  """
  def __init__(self, metric="euclidean", **kwargs):
    GenericMethod.__init__(
      self,
      losses.EnergyLoss(),
      transformers.DistanceTransformer(metric),
      **kwargs
    )

class EDy(GenericMethod):
  """The energy distance-based EDy method by Castaño et al. (2022).

  This subclass of `GenericMethod` is instantiated with an `EnergyLoss` and a `DistanceTransformer`, the latter of which uses a `ClassTransformer` as a preprocessor.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      metric (optional): The metric with which the distance between data items is measured. Can take any value that is accepted by `scipy.spatial.distance.cdist`. Defaults to `"euclidean"`.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      **kwargs: Keyword arguments accepted by `GenericMethod`.
  """
  def __init__(self, classifier, metric="euclidean", fit_classifier=True, **kwargs):
    GenericMethod.__init__(
      self,
      losses.EnergyLoss(),
      transformers.DistanceTransformer(
        metric,
        preprocessor = transformers.ClassTransformer(
          classifier,
          fit_classifier = fit_classifier,
          is_probabilistic = True,
        )
      ),
      **kwargs
    )

class HDx(GenericMethod):
  """The Hellinger distance-based HDx method by González-Castro et al. (2013).

  This subclass of `GenericMethod` is instantiated with a `HellingerSurrogateLoss` and a `HistogramTransformer`.

  Args:
      n_bins: The number of bins in each feature.
      **kwargs: Keyword arguments accepted by `GenericMethod`.
  """
  def __init__(self, n_bins, **kwargs):
    GenericMethod.__init__(
      self,
      losses.HellingerSurrogateLoss(),
      transformers.HistogramTransformer(n_bins, unit_scale=False),
      **kwargs
    )

class HDy(GenericMethod):
  """The Hellinger distance-based HDy method by González-Castro et al. (2013).

  This subclass of `GenericMethod` is instantiated with a `HellingerSurrogateLoss` and a `HistogramTransformer`, the latter of which uses a `ClassTransformer` as a preprocessor.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      n_bins: The number of bins in each class.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      **kwargs: Keyword arguments accepted by `GenericMethod`.
  """
  def __init__(self, classifier, n_bins, *, fit_classifier=True, **kwargs):
    GenericMethod.__init__(
      self,
      losses.HellingerSurrogateLoss(),
      transformers.HistogramTransformer(
        n_bins,
        preprocessor = transformers.ClassTransformer(
          classifier,
          fit_classifier = fit_classifier,
          is_probabilistic = True,
        ),
        unit_scale = False,
      ),
      **kwargs
    )

class KMM(GenericMethod):
  """The kernel-based KMM method by Dussap et al. (2023).
  
  This subclass of `GenericMethod` is instantiated with a `LeastSquaresLoss` and a `KernelTransformer`.

  Args:
      kernel (optional): Which kernel to use. Can be a callable with the signature `(X[y==i], X[y==j]) -> scalar` or one of "energy", "gaussian", and "laplacian". Defaults to "energy".
      sigma (optional): A smoothing parameter in the `gaussian` and `laplacian` kernels. Defaults to `1`.
      **kwargs: Keyword arguments accepted by `GenericMethod`.
  """
  def __init__(self, kernel="energy", sigma=1, **kwargs):
    if kernel == "energy":
      transformer = transformers.EnergyKernelTransformer()
    elif kernel == "gaussian":
      transformer = transformers.GaussianKernelTransformer(sigma=sigma)
    elif kernel == "laplacian":
      transformer = transformers.LaplacianKernelTransformer(sigma=sigma)
    else:
      transformer = transformers.KernelTransformer(kernel)
    GenericMethod.__init__(self, losses.LeastSquaresLoss(), transformer, **kwargs)
