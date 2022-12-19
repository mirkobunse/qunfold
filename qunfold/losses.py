import numpy as np
import sympy
from scipy import optimize

class Result:
  def __init__(self, loss, method="trust-exact"):
    l_0 = np.random.rand(loss.dim)
    opt = optimize.minimize(loss.fun, l_0, jac=loss.jac, hess=loss.hess, method=method)
    self.p_est = loss.post(opt.x)
    self.iterations = opt.nit
    self.message = opt.message

# helper function for our softmax "trick"
def _softmax(l):
  exp_l = l.applyfunc(sympy.exp)
  sum_exp_l = 0
  for i in range(len(exp_l)):
    sum_exp_l += exp_l[i]
  return exp_l / sum_exp_l

# helper function for least squares
def _lsq(q, M, p):
  v = q - M*p
  return v.transpose()*v

class LeastSquaresLoss:
  """The loss function of ACC, PACC, and ReadMe."""
  def __init__(self, q, M, softmax=True):
    """Assemble the loss function and its derivatives."""
    n_features, n_classes = M.shape
    q = sympy.ImmutableMatrix(q.reshape(n_features, 1))
    M = sympy.ImmutableMatrix(M)
    if softmax:
      l = sympy.MatrixSymbol("l", n_classes-1, 1)
      p = _softmax(sympy.Matrix([[0]]).row_insert(1, l)) # softmax with l[0]=0
      reshape_l = lambda l_i: l_i.reshape(n_classes-1, 1)
      post = lambda l_i: np.exp(np.concatenate(([0.], l_i))) / (1 + np.exp(l_i).sum())
    else:
      p = sympy.MatrixSymbol("p", n_classes, 1)
      l = p
      reshape_l = lambda l_i: l_i.reshape(n_classes, 1)
      post = lambda l_i: l_i / l_i.sum()
    L = _lsq(q, M, p).as_explicit()
    J = L.jacobian(l)
    fun = sympy.lambdify(l, sympy.flatten(L.evalf()))
    jac = sympy.lambdify(l, sympy.flatten(J.transpose())) # gradient vector
    hess = sympy.lambdify(l, J.jacobian(l)) # Hessian matrix
    self.fun = lambda l_i: fun(reshape_l(l_i))[0]
    self.jac = lambda l_i: np.array(jac(reshape_l(l_i)))
    self.hess = lambda l_i: np.array(hess(reshape_l(l_i)))
    self.post = post
    self.dim = n_classes - (1 if softmax else 0)
