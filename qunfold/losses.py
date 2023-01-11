import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

# helper function for our softmax "trick" with l[0]=0
def _jnp_softmax(l):
  exp_l = jnp.exp(l)
  return jnp.concatenate((jnp.ones(1), exp_l)) / (1. + exp_l.sum())

# helper function for least squares
def _lsq(q, M, p):
  v = q - jnp.dot(M, p)
  return jnp.dot(v, v)

def instantiate_loss(loss, q, M):
  """Create a dict of JAX functions "fun", "jac", and "hess" of l."""
  q = jnp.array(q)
  M = jnp.array(M)
  fun = lambda l: loss._instantiate(q, M)(_jnp_softmax(l)) # loss function
  jac = jax.grad(fun)
  hess = jax.jacfwd(jac) # forward-mode AD
  return {"fun": fun, "jac": jac, "hess": hess}

class AbstractLoss(ABC):
  """Abstract base class for loss functions."""
  @abstractmethod
  def _instantiate(self, q, M):
    """Create a JAX function p -> loss."""
    pass

class LeastSquaresLoss(AbstractLoss):
  """The loss function of ACC, PACC, and ReadMe."""
  def _instantiate(self, q, M):
    return lambda p: _lsq(q, M, p)

def _combine_losses(losses, weights, q, M, p):
  combined_loss = 0
  for (loss, weight) in (losses, weights):
    combined_loss += weight * loss._instantiate(q, M)(p)
  return combined_loss

class CombinedLoss(AbstractLoss):
  """The weighted sum of multiple losses."""
  def __init__(self, *losses, weights=None):
    self.losses = losses
    if weights is None:
      weights = jnp.ones(len(losses))
    self.weights = weights
  def _instantiate(self, q, M):
    return lambda p: _combine_losses(self.losses, self.weights, q, M, p)
