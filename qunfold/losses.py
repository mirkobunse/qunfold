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

# helper function for RUN's maximum likelihood loss
def _blobel(q, M, p, N):
  Mp = jnp.dot(M, N * p)
  return jnp.sum(Mp - N * q * jnp.log(Mp))

# helper function for Boolean masks M[_nonzero_features(M),:] and q[_nonzero_features(M)]
def _nonzero_features(M):
  return jnp.any(M != 0, axis=1)

def instantiate_loss(loss, q, M, N):
  """Create a dict of JAX functions "fun", "jac", and "hess" of l."""
  q = jnp.array(q)
  M = jnp.array(M)
  fun = lambda l: loss._instantiate(q, M, N)(_jnp_softmax(l)) # loss function
  jac = jax.grad(fun)
  hess = jax.jacfwd(jac) # forward-mode AD
  return {"fun": fun, "jac": jac, "hess": hess}

class AbstractLoss(ABC):
  """Abstract base class for loss functions."""
  @abstractmethod
  def _instantiate(self, q, M, N):
    """Create a JAX function p -> loss."""
    pass

class LeastSquaresLoss(AbstractLoss):
  """The loss function of ACC, PACC, and ReadMe."""
  def _instantiate(self, q, M, N):
    nonzero = _nonzero_features(M)
    M = M[nonzero,:]
    q = q[nonzero]
    return lambda p: _lsq(q, M, p)

class BlobelLoss(AbstractLoss):
  """The loss function of RUN by Blobel (1985)."""
  def _instantiate(self, q, M, N):
    if N is None:
      raise ValueError("BlobelLoss does not allow N=None")
    nonzero = _nonzero_features(M)
    M = M[nonzero,:]
    q = q[nonzero]
    return lambda p: _blobel(q, M, p, N)

# helper function for CombinedLoss
def _combine_losses(losses, weights, q, M, p, N):
  combined_loss = 0
  for (loss, weight) in zip(losses, weights):
    combined_loss += weight * loss._instantiate(q, M, N)(p)
  return combined_loss

class CombinedLoss(AbstractLoss):
  """The weighted sum of multiple losses."""
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
  return jnp.dot(Tp, Tp)

class TikhonovRegularization(AbstractLoss):
  """Tikhonov regularization, as proposed by Blobel (1985)."""
  def _instantiate(self, q, M, N):
    T = _tikhonov_matrix(M.shape[1])
    return lambda p: _tikhonov(p, T)

def TikhonovRegularized(loss, tau=0.):
  """Convenience function to add TikhonovRegularization to a loss."""
  return CombinedLoss(loss, TikhonovRegularization(), weights=[1, tau])
