import numpy as np
import traceback
from . import (losses, representations)
from .. import AbstractMethod, minimize

class LinearMethod(AbstractMethod):
  """A generic quantification / unfolding method that predicts class prevalences by solving a system of linear equations.

  This class represents any method that consists of a loss function, a data representation, and a regularization term. In this implementation, any regularized loss is minimized through unconstrained second-order minimization. Valid probability estimates are ensured through a soft-max trick by Bunse (2022).

  Args:
      loss: An instance from `qunfold.methods.linear.losses`.
      representation: An instance from `qunfold.methods.linear.representations`.
      solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to `"trust-ncg"`.
      solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `{"gtol": 1e-8, "maxiter": 1000}`.
      seed (optional): A random number generator seed from which a numpy RandomState is created. Defaults to `None`.

  Examples:
      Here, we create the ordinal variant of ACC (Bunse et al., 2023). This variant consists of the original data representation of ACC and of the original loss of ACC, the latter of which is regularized towards smooth solutions.

          >>> LinearMethod(
          >>>     TikhonovRegularized(LeastSquaresLoss(), 0.01),
          >>>     ClassRepresentation(RandomForestClassifier(oob_score=True))
          >>> )
  """
  def __init__(self, loss, representation,
      solver = "trust-ncg",
      solver_options = {"gtol": 1e-8, "maxiter": 1000},
      seed = None,
      ):
    self.loss = loss
    self.representation = representation
    self.solver = solver
    self.solver_options = solver_options
    self.seed = seed
  def fit(self, X, y, n_classes=None):
    self.M = self.representation.fit_transform(X, y, n_classes=n_classes)
    return self
  def predict(self, X):
    q = self.representation.transform(X)
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
    return minimize(
      self.loss.instantiate(q, M, N),
      M.shape[1], # = n_classes
      np.random.RandomState(self.seed), # = rng
      self.solver,
      self.solver_options
    )
  @property
  def p_trn(self):
    return self.representation.p_trn

class ACC(LinearMethod):
  """Adjusted Classify & Count by Forman (2008).

  This subclass of `LinearMethod` is instantiated with a `LeastSquaresLoss` and a `ClassRepresentation`.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, classifier, fit_classifier=True, **kwargs):
    LinearMethod.__init__(
      self,
      losses.LeastSquaresLoss(),
      representations.ClassRepresentation(
        classifier,
        fit_classifier = fit_classifier
      ),
      **kwargs
    )

class PACC(LinearMethod):
  """Probabilistic Adjusted Classify & Count by Bella et al. (2010).

  This subclass of `LinearMethod` is instantiated with a `LeastSquaresLoss` and a `ClassRepresentation`.

  Args:
      classifier: A classifier that implements the API of scikit-learn.
      fit_classifier (optional): Whether to fit the `classifier` when this quantifier is fitted. Defaults to `True`.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, classifier, fit_classifier=True, **kwargs):
    LinearMethod.__init__(
      self,
      losses.LeastSquaresLoss(),
      representations.ClassRepresentation(
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
      representation: An instance from `qunfold.methods.linear.representations`.
      tau (optional): The regularization strength. Defaults to 0.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, representation, *, tau=0., **kwargs):
    LinearMethod.__init__(
      self,
      losses.TikhonovRegularized(losses.BlobelLoss(), tau),
      representation,
      **kwargs
    )

class EDx(LinearMethod):
  """The energy distance-based EDx method by Kawakubo et al. (2016).

  This subclass of `LinearMethod` is instantiated with an `EnergyLoss` and a `DistanceRepresentation`.

  Args:
      metric (optional): The metric with which the distance between data items is measured. Can take any value that is accepted by `scipy.spatial.distance.cdist`. Defaults to `"euclidean"`.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, metric="euclidean", **kwargs):
    LinearMethod.__init__(
      self,
      losses.EnergyLoss(),
      representations.DistanceRepresentation(metric),
      **kwargs
    )

class EDy(LinearMethod):
  """The energy distance-based EDy method by Castaño et al. (2022).

  This subclass of `LinearMethod` is instantiated with an `EnergyLoss` and a `DistanceRepresentation`, the latter of which uses a `ClassRepresentation` as a preprocessor.

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
      representations.DistanceRepresentation(
        metric,
        preprocessor = representations.ClassRepresentation(
          classifier,
          fit_classifier = fit_classifier,
          is_probabilistic = True,
        )
      ),
      **kwargs
    )

class HDx(LinearMethod):
  """The Hellinger distance-based HDx method by González-Castro et al. (2013).

  This subclass of `LinearMethod` is instantiated with a `HellingerSurrogateLoss` and a `HistogramRepresentation`.

  Args:
      n_bins: The number of bins in each feature.
      **kwargs: Keyword arguments accepted by `LinearMethod`.
  """
  def __init__(self, n_bins, **kwargs):
    LinearMethod.__init__(
      self,
      losses.HellingerSurrogateLoss(),
      representations.HistogramRepresentation(n_bins, unit_scale=False),
      **kwargs
    )

class HDy(LinearMethod):
  """The Hellinger distance-based HDy method by González-Castro et al. (2013).

  This subclass of `LinearMethod` is instantiated with a `HellingerSurrogateLoss` and a `HistogramRepresentation`, the latter of which uses a `ClassRepresentation` as a preprocessor.

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
      representations.HistogramRepresentation(
        n_bins,
        preprocessor = representations.ClassRepresentation(
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

  This subclass of `LinearMethod` is instantiated with a `LeastSquaresLoss` and an instance of a `KernelRepresentation` sub-class that corresponds to the `kernel` argument.

  Args:
      kernel (optional): Which kernel to use. Can be a callable with the signature `(X[y==i], X[y==j]) -> scalar` or one of "energy", "gaussian", "laplacian" and "rff". Defaults to "energy".
      sigma (optional): A smoothing parameter that is used if `kernel in ["gaussian", "laplacian", "rff"]`. Defaults to `1`.
      n_rff (optional): The number of random Fourier features if `kernel == "rff"`. Defaults to `1000`.
      **kwargs: Keyword arguments accepted by `LinearMethod`. The `seed` argument also controls the randomness of the random Fourier features if `kernel == "rff"`.
  """
  def __init__(self, kernel="energy", sigma=1, n_rff=1000, seed=None, **kwargs):
    if kernel == "energy":
      representation = representations.EnergyKernelRepresentation()
    elif kernel == "gaussian":
      representation = representations.GaussianKernelRepresentation(sigma=sigma)
    elif kernel == "laplacian":
      representation = representations.LaplacianKernelRepresentation(sigma=sigma)
    elif kernel == "rff":
      representation = representations.GaussianRFFKernelRepresentation(
        sigma = sigma,
        n_rff = n_rff,
        seed = seed,
      )
    else:
      representation = representations.KernelRepresentation(kernel)
    LinearMethod.__init__(self, losses.LeastSquaresLoss(), representation, seed=seed, **kwargs)
