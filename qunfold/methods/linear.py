import numpy as np
import traceback
from scipy.optimize import minimize
from . import (
  AbstractMethod,
  rand_x0,
  np_softmax,
  Result,
  DerivativeError,
  check_derivative,
  MinimizeCallbackState
)
from .. import (losses, transformers)

class LinearMethod(AbstractMethod):
  """A generic quantification / unfolding method that solves a linear system of equations.

  This class represents any method that consists of a loss function, a feature transformation, and a regularization term. In this implementation, any regularized loss is minimized through unconstrained second-order minimization. Valid probability estimates are ensured through a soft-max trick by Bunse (2022).

  Args:
      loss: An instance from `qunfold.losses`.
      transformer: An instance from `qunfold.transformers`.
      solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to `"trust-ncg"`.
      solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `{"gtol": 1e-8, "maxiter": 1000}`.
      seed (optional): A random number generator seed from which a numpy RandomState is created. Defaults to `None`.

  Examples:
      Here, we create the ordinal variant of ACC (Bunse et al., 2023). This variant consists of the original feature transformation of ACC and of the original loss of ACC, the latter of which is regularized towards smooth solutions.

          >>> LinearMethod(
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
  def fit(self, X, y, n_classes=None):
    self.M = self.transformer.fit_transform(X, y, n_classes=n_classes)
    return self
  def predict(self, X):
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
    x0 = rand_x0(rng, M.shape[1]) # random starting point
    state = MinimizeCallbackState(x0)
    try:
      opt = minimize(
        loss_dict["fun"], # JAX function l -> loss
        x0,
        jac = check_derivative(loss_dict["jac"], "jac"),
        hess = check_derivative(loss_dict["hess"], "hess"),
        method = self.solver,
        options = self.solver_options,
        callback = state.callback()
      )
    except (DerivativeError, ValueError):
      traceback.print_exc()
      opt = state.get_state()
    return Result(np_softmax(opt.x), opt.nit, opt.message)
  @property
  def p_trn(self):
    return self.transformer.p_trn

class ACC(LinearMethod):
  """Adjusted Classify & Count by Forman (2008).

  This subclass of `LinearMethod` is instantiated with a `LeastSquaresLoss` and a `ClassTransformer`.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, classifier, fit_classifier=True, **kwargs):
    LinearMethod.__init__(
      self,
      losses.LeastSquaresLoss(),
      transformers.ClassTransformer(
        classifier,
        fit_classifier = fit_classifier
      ),
      **kwargs
    )

class PACC(LinearMethod):
  """Probabilistic Adjusted Classify & Count by Bella et al. (2010).

  This subclass of `LinearMethod` is instantiated with a `LeastSquaresLoss` and a `ClassTransformer`.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, classifier, fit_classifier=True, **kwargs):
    LinearMethod.__init__(
      self,
      losses.LeastSquaresLoss(),
      transformers.ClassTransformer(
        classifier,
        fit_classifier = fit_classifier,
        is_probabilistic = True
      ),
      **kwargs
    )

class RUN(LinearMethod):
  """Regularized Unfolding by Blobel (1985).

  This subclass of `LinearMethod` is instantiated with a `TikhonovRegularized(BlobelLoss)`.

  Args:
      transformer: An instance from `qunfold.transformers`.
      tau (optional): The regularization strength. Defaults to 0.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, transformer, *, tau=0., **kwargs):
    LinearMethod.__init__(
      self,
      losses.TikhonovRegularized(losses.BlobelLoss(), tau),
      transformer,
      **kwargs
    )

class EDx(LinearMethod):
  """The energy distance-based EDx method by Kawakubo et al. (2016).

  This subclass of `LinearMethod` is instantiated with an `EnergyLoss` and a `DistanceTransformer`.

  Args:
      metric (optional): The metric with which the distance between data items is measured. Can take any value that is accepted by `scipy.spatial.distance.cdist`. Defaults to `"euclidean"`.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, metric="euclidean", **kwargs):
    LinearMethod.__init__(
      self,
      losses.EnergyLoss(),
      transformers.DistanceTransformer(metric),
      **kwargs
    )

class EDy(LinearMethod):
  """The energy distance-based EDy method by Castaño et al. (2022).

  This subclass of `LinearMethod` is instantiated with an `EnergyLoss` and a `DistanceTransformer`, the latter of which uses a `ClassTransformer` as a preprocessor.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      metric (optional): The metric with which the distance between data items is measured. Can take any value that is accepted by `scipy.spatial.distance.cdist`. Defaults to `"euclidean"`.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, classifier, metric="euclidean", fit_classifier=True, **kwargs):
    LinearMethod.__init__(
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

class HDx(LinearMethod):
  """The Hellinger distance-based HDx method by González-Castro et al. (2013).

  This subclass of `LinearMethod` is instantiated with a `HellingerSurrogateLoss` and a `HistogramTransformer`.

  Args:
      n_bins: The number of bins in each feature.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, n_bins, **kwargs):
    LinearMethod.__init__(
      self,
      losses.HellingerSurrogateLoss(),
      transformers.HistogramTransformer(n_bins, unit_scale=False),
      **kwargs
    )

class HDy(LinearMethod):
  """The Hellinger distance-based HDy method by González-Castro et al. (2013).

  This subclass of `LinearMethod` is instantiated with a `HellingerSurrogateLoss` and a `HistogramTransformer`, the latter of which uses a `ClassTransformer` as a preprocessor.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      n_bins: The number of bins in each class.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, classifier, n_bins, *, fit_classifier=True, **kwargs):
    LinearMethod.__init__(
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

class KMM(LinearMethod):
  """The kernel-based KMM method with random Fourier features by Dussap et al. (2023).

  This subclass of `LinearMethod` is instantiated with a `LeastSquaresLoss` and an instance of a `KernelTransformer` sub-class that corresponds to the `kernel` argument.

  Args:
      kernel (optional): Which kernel to use. Can be a callable with the signature `(X[y==i], X[y==j]) -> scalar` or one of "energy", "gaussian", "laplacian" and "rff". Defaults to "energy".
      sigma (optional): A smoothing parameter that is used if `kernel in ["gaussian", "laplacian", "rff"]`. Defaults to `1`.
      n_rff (optional): The number of random Fourier features if `kernel == "rff"`. Defaults to `1000`.
      **kwargs: Keyword arguments accepted by `LinearMethod`. The `seed` argument also controls the randomness of the random Fourier features if `kernel == "rff"`.
  """
  def __init__(self, kernel="energy", sigma=1, n_rff=1000, seed=None, **kwargs):
    if kernel == "energy":
      transformer = transformers.EnergyKernelTransformer()
    elif kernel == "gaussian":
      transformer = transformers.GaussianKernelTransformer(sigma=sigma)
    elif kernel == "laplacian":
      transformer = transformers.LaplacianKernelTransformer(sigma=sigma)
    elif kernel == "rff":
      transformer = transformers.GaussianRFFKernelTransformer(
        sigma = sigma,
        n_rff = n_rff,
        seed = seed,
      )
    else:
      transformer = transformers.KernelTransformer(kernel)
    LinearMethod.__init__(self, losses.LeastSquaresLoss(), transformer, seed=seed, **kwargs)

class KDEyHD(LinearMethod):
  """The kernel-based KDE method with Monte Carlo sampling by González-Moreo et al. (2024).

   This subclass of `LinearMethod` is instantiated with a `KDEyHDLoss` and a `KDEyHDTransformer`.

   Args:
      classifier: A classifier that implements the API of scikit-learn.
      bandwith: A smoothing parameter for the kernel-function.
      random_state (optional): Controls the randomness of the Monte Carlo sampling. Defaults to `0`.
      n_trials (optional): The number of Monte Carlo samples to generate. Defaults to `10_000`.
  """
  def __init__(self, classifier, bandwidth, random_state=0, n_trials=10_000, **kwargs):
    LinearMethod.__init__(
      self,
      losses.KDEyHDLoss(),
      transformers.KDEyHDTransformer(
        kernel="gaussian",
        bandwidth=bandwidth,
        classifier=classifier,
        random_state=random_state,
        n_trials=n_trials
      ),
      **kwargs
    )

class KDEyCS(LinearMethod):
  """A closed-form solution of the kernel-based KDE method by González-Moreo et al. (2024).

   This subclass of `LinearMethod` is instantiated with a `KDEyCSLoss` and a `KDEyCSTransformer`.

   Args:
      classifier: A classifier that implements the API of scikit-learn.
      bandwith: A smoothing parameter for the kernel-function.
      y_trn: The class labels of the training dataset.
  """
  def __init__(self, classifier, bandwidth, **kwargs):
    LinearMethod.__init__(
      self,
      losses.KDEyCSLoss(),
      transformers.KDEyCSTransformer(
        kernel="gaussian",
        bandwidth=bandwidth,
        classifier=classifier
      ),
      **kwargs
    )
  def fit(self, X, y, n_classes=None):
    """Fit this quantifier to data.

    Args:
        X: The feature matrix to which this quantifier will be fitted.
        y: The labels to which this quantifier will be fitted.
        n_classes (optional): The number of expected classes. Defaults to `None`.

    Returns:
        This fitted quantifier itself.
    """
    self.M = self.transformer.fit_transform(X, y, n_classes=n_classes)
    self.loss.counts_inv = self.transformer.counts_inv
    return self

class KDEyMLID(LinearMethod):
  """The Maximum-Likelihood solution of the kernel-based KDE method by González-Moreo et al. (2024).

   This subclass of `LinearMethod` is instantiated with a `KDEyMLLoss` and a `KDEyMLTransformer`.

   Args:
      classifier: A classifier that implements the API of scikit-learn.
      bandwith: A smoothing parameter for the kernel-function.
  """
  def __init__(self, classifier, bandwidth, **kwargs):
    LinearMethod.__init__(
      self,
      losses.KDEyMLLoss(),
      transformers.KDEyMLTransformerID(
        kernel="gaussian",
        bandwidth=bandwidth,
        classifier=classifier
      ),
      **kwargs
    )
