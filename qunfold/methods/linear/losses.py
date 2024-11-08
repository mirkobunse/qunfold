import jax
import jax.numpy as jnp
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from ...base import BaseMixin

# helper function for least squares
def _lsq(p, q, M, N=None):
  v = q - jnp.dot(M, p)
  return jnp.dot(v, v)

# helper function for RUN's maximum likelihood loss
def _blobel(p, q, M, N):
  Mp = jnp.dot(M, N * p)
  return jnp.sum(Mp - N * q * jnp.log(Mp))

# helper function for the energy distance-based loss
def _energy(p, q, M, N=None):
  return jnp.dot(p, 2 * q - jnp.dot(M, p))

# helper function for the Hellinger surrogate loss, leveraging the fact that the
# average of squared distances can be computed over a single concatenation of histograms
def _hellinger_surrogate(p, q, M, N=None):
  i = jnp.logical_and(q > 0, jnp.any(M > 0, axis=1)) # ignore constant zeros to avoid NaNs
  return -jnp.sqrt(q[i] * jnp.dot(M[i], p)).sum()

# helper function for Boolean masks M[_nonzero_features(M),:] and q[_nonzero_features(M)]
def _nonzero_features(M):
  return jnp.any(M != 0, axis=1)

class AbstractLoss(ABC,BaseMixin):
  """Abstract base class for loss functions and for regularization terms."""
  @abstractmethod
  def instantiate(self, q, M, N):
    """This abstract method has to create a lambda expression `p -> loss` with JAX.

    In particular, your implementation of this abstract method should return a lambda expression

        >>> return lambda p: loss_value(q, M, p, N)

    where `loss_value` has to return the result of a JAX expression. The JAX requirement ensures that the loss function can be auto-differentiated. Hence, no derivatives of the loss function have to be provided manually. JAX expressions are easy to implement. Just import the numpy wrapper

        >>> import jax.numpy as jnp

    and use `jnp` just as if you would use numpy.

    Note:
        `p` is a vector of class-wise probabilities. This vector will already be the result of our soft-max trick, so that you don't have to worry about constraints or latent parameters.

    Args:
        q: A numpy array.
        M: A numpy matrix.
        N: The number of data items that `q` represents.

    Returns:
        A lambda expression `p -> loss`, implemented in JAX.

    Examples:
        The least squares loss, `(q - M*p)' * (q - M*p)`, is simply

            >>> jnp.dot(q - jnp.dot(M, p), q - jnp.dot(M, p))
    """
    pass

@dataclass
class FunctionLoss(AbstractLoss):
  """Create a loss object from a JAX function `(p, q, M, N) -> loss_value`.

  Using this class is likely more convenient than subtyping *AbstractLoss*. In both cases, the `loss_value` has to be the result of a JAX expression. The JAX requirement ensures that the loss function can be auto-differentiated. Hence, no derivatives of the loss function have to be provided manually. JAX expressions are easy to implement. Just import the numpy wrapper

      >>> import jax.numpy as jnp

  and use `jnp` just as if you would use numpy.

  Note:
      `p` is a vector of class-wise probabilities. This vector will already be the result of our soft-max trick, so that you don't have to worry about constraints or latent parameters.

  Args:
      loss_function: A JAX function `(p, q, M, N) -> loss_value`.

  Examples:
      The least squares loss, `(q - M*p)' * (q - M*p)`, is simply

          >>> def least_squares(p, q, M, N):
          >>>     jnp.dot(q - jnp.dot(M, p), q - jnp.dot(M, p))

      and thereby ready to be used in a *FunctionLoss* object:

          >>> least_squares_loss = FunctionLoss(least_squares)
  """
  loss_function: Callable
  def instantiate(self, q, M, N=None):
    nonzero = _nonzero_features(M)
    M = jnp.array(M)[nonzero,:]
    q = jnp.array(q)[nonzero]
    return lambda p: self.loss_function(p, q, M, N)

class LeastSquaresLoss(FunctionLoss):
  """The loss function of ACC (Forman, 2008), PACC (Bella et al., 2019), and ReadMe (Hopkins & King, 2010).

  This loss function computes the sum of squares of element-wise errors between `q` and `M*p`.
  """
  def __init__(self):
    super().__init__(_lsq)

class BlobelLoss(FunctionLoss):
  """The loss function of RUN (Blobel, 1985).

  This loss function models a likelihood function under the assumption of independent Poisson-distributed elements of `q` with Poisson rates `M*p`.
  """
  def __init__(self):
    super().__init__(_blobel)

class EnergyLoss(FunctionLoss):
  """The loss function of EDx (Kawakubo et al., 2016) and EDy (Castaño et al., 2022).

  This loss function represents the Energy Distance between two samples.
  """
  def __init__(self):
    super().__init__(_energy)

class HellingerSurrogateLoss(FunctionLoss):
  """The loss function of HDx and HDy (González-Castro et al., 2013).

  This loss function computes the average of the squared Hellinger distances between feature-wise (or class-wise) histograms. Note that the original HDx and HDy by González-Castro et al (2013) do not use the squared but the regular Hellinger distance. Their approach is problematic because the regular distance is not always twice differentiable and, hence, complicates numerical optimizations.
  """
  def __init__(self):
    super().__init__(_hellinger_surrogate)    


# helper function for CombinedLoss
def _combine_losses(losses, weights, q, M, p, N):
  combined_loss = 0
  for (loss, weight) in zip(losses, weights):
    combined_loss += weight * loss.instantiate(q, M, N)(p)
  return combined_loss

class CombinedLoss(AbstractLoss):
  """The weighted sum of multiple losses.

  Args:
      *losses: An arbitrary number of losses to be added together.
      weights (optional): An array of weights which the losses are scaled.
  """
  def __init__(self, *losses, weights=None):
    self.losses = losses
    self.weights = weights
  def instantiate(self, q, M, N):
    weights = self.weights
    if weights is None:
      weights = jnp.ones(len(self.losses))
    return lambda p: _combine_losses(self.losses, self.weights, q, M, p, N)

# helpers for TikhonovRegularization
def _tikhonov_matrix(C):
  return (
    jnp.diag(jnp.full(C, 2))
    + jnp.diag(jnp.full(C-1, -1), -1)
    + jnp.diag(jnp.full(C-1, -1), 1)
  )[1:C-1, :]
def _tikhonov(p, T):
  Tp = jnp.dot(T, p)
  return jnp.dot(Tp, Tp) / 2

class TikhonovRegularization(AbstractLoss):
  """Tikhonov regularization, as proposed by Blobel (1985).

  This regularization promotes smooth solutions. This behavior is often required in ordinal quantification and in unfolding problems.
  """
  def instantiate(self, q, M, N):
    T = _tikhonov_matrix(M.shape[1])
    return lambda p: _tikhonov(p, T)

# TikhonovRegularized is implemented as a function instead of a class to facilitate
# the inspection that the QuaPyWrapper takes out.

def TikhonovRegularized(loss, tau=0.):
  """Add TikhonovRegularization (Blobel, 1985) to any loss.

  Calling this function is equivalent to calling

      >>> CombinedLoss(loss, TikhonovRegularization(), weights=[1, tau])

  Args:
      loss: An instance from `qunfold.losses`.
      tau (optional): The regularization strength. Defaults to 0.

  Returns:
      An instance of `CombinedLoss`.

  Examples:
      The regularized loss of RUN (Blobel, 1985) is:

          >>> TikhonovRegularization(BlobelLoss(), tau)
  """
  return CombinedLoss(loss, TikhonovRegularization(), weights=[1, tau])
