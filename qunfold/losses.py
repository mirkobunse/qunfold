import numpy as np
import sympy
from abc import ABC, abstractmethod

# helper function for our softmax "trick" with l[0]=0
def _sympy_softmax(l):
  exp_l = sympy.Matrix([[0]]).row_insert(1, l).applyfunc(sympy.exp)
  sum_exp_l = 0
  for i in range(len(exp_l)):
    sum_exp_l += exp_l[i]
  return exp_l / sum_exp_l

# helper function for least squares
def _lsq(q, M, p):
  v = q - M*p
  return v.transpose()*v

def instantiate_loss(loss, q, M):
  """Create a dict of fun, jac, hess, and n_classes."""
  n_features, n_classes = M.shape
  q = sympy.ImmutableMatrix(q.reshape(n_features, 1))
  M = sympy.ImmutableMatrix(M)
  l = sympy.MatrixSymbol("l", n_classes-1, 1) # latent variables
  L = loss._instantiate(q, M, _sympy_softmax(l)).as_explicit() # loss
  J = L.jacobian(l)
  fun = sympy.lambdify(l, sympy.flatten(L.evalf()))
  jac = sympy.lambdify(l, sympy.flatten(J.transpose())) # gradient vector
  hess = sympy.lambdify(l, J.jacobian(l)) # Hessian matrix
  return {
    "fun": lambda _l: fun(_l.reshape(n_classes-1, 1))[0],
    "jac": lambda _l: np.array(jac(_l.reshape(n_classes-1, 1))),
    "hess": lambda _l: np.array(hess(_l.reshape(n_classes-1, 1))),
    "n_classes": n_classes, 
  }

class AbstractLoss(ABC):
  """Abstract base class for loss functions."""
  @abstractmethod
  def _instantiate(self, q, M, p):
    """Create a sympy representation of this loss function."""
    pass

class LeastSquaresLoss(AbstractLoss):
  """The loss function of ACC, PACC, and ReadMe."""
  def _instantiate(self, q, M, p):
    return _lsq(q, M, p)

class CombinedLoss(AbstractLoss):
  """The weighted sum of multiple losses."""
  def __init__(self, *losses, weights=None):
    self.losses = losses
    if weights is None:
      weights = np.ones(len(losses))
    self.weights = weights
  def _instantiate(self, q, M):
    combined_loss = 0
    for (loss, weight) in (self.losses, self.weights):
      combined_loss += weight * loss._instantiate(q, M, _sympy_softmax(self.l))
    return combined_loss
