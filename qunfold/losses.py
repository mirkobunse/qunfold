import jax
import jax.numpy as jnp
import functools
from abc import ABC, abstractmethod

# helper function for our softmax "trick" with l[0]=0
def _jnp_softmax(l):
  exp_l = jnp.exp(l)
  return jnp.concatenate((jnp.ones(1), exp_l)) / (1. + exp_l.sum())

# helper function for least squares
def _lsq(p, q, M, N=None):
  v = q - jnp.dot(M, p)
  return jnp.dot(v, v)

# helper function for RUN's maximum likelihood loss
def _blobel(p, q, M, N):
  Mp = jnp.dot(M, N * p)
  return jnp.sum(Mp - N * q * jnp.log(Mp))

# helper function for Boolean masks M[_nonzero_features(M),:] and q[_nonzero_features(M)]
def _nonzero_features(M):
  return jnp.any(M != 0, axis=1)

def instantiate_loss(loss, q, M, N):
  """Create a dict of JAX functions "fun", "jac", and "hess" of the loss."""
  q = jnp.array(q)
  M = jnp.array(M)
  fun = lambda l: loss._instantiate(q, M, N)(_jnp_softmax(l)) # loss function
  jac = jax.grad(fun)
  hess = jax.jacfwd(jac) # forward-mode AD
  return {"fun": fun, "jac": jac, "hess": hess}

class AbstractLoss(ABC):
  """Abstract base class for loss functions and for regularization terms."""
  @abstractmethod
  def _instantiate(self, q, M, N):
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
  def __init__(self, loss_function):
    self.loss_function = loss_function
  def _instantiate(self, q, M, N=None):
    nonzero = _nonzero_features(M)
    M = M[nonzero,:]
    q = q[nonzero]
    return lambda p: self.loss_function(p, q, M, N)

class LeastSquaresLoss(FunctionLoss):
  """The loss function of ACC, PACC, and ReadMe.

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

# helper function for CombinedLoss
def _combine_losses(losses, weights, q, M, p, N):
  combined_loss = 0
  for (loss, weight) in zip(losses, weights):
    combined_loss += weight * loss._instantiate(q, M, N)(p)
  return combined_loss

class CombinedLoss(AbstractLoss):
  """The weighted sum of multiple losses.

  Args:
      *losses: An arbitrary number of losses to be added together.
      weights (optional): An array of weights which the losses are scaled.
  """
  def __init__(self, *losses, weights=None):
    self.losses = losses
    if weights is None:
      weights = jnp.ones(len(losses))
    self.weights = weights
  def _instantiate(self, q, M, N):
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
  def _instantiate(self, q, M, N):
    T = _tikhonov_matrix(M.shape[1])
    return lambda p: _tikhonov(p, T)

class TikhonovRegularized(CombinedLoss):
  """Add TikhonovRegularization to any loss.

  Instances of this class are equivalent to

      >>> CombinedLoss(loss, TikhonovRegularization(), weights=[1, tau])

  Args:
      loss: An instance from `qunfold.losses`.
      tau (optional): The regularization strength. Defaults to 0.

  Examples:
      The regularized loss of RUN (Blobel, 1985) is:

          >>> TikhonovRegularization(BlobelLoss(), tau)
  """
  def __init__(self, loss, tau=0.):
    super().__init__(loss, TikhonovRegularization(), weights=[1, tau])
