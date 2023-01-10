import numpy as np
from scipy import optimize
from . import losses

# helper function for our softmax "trick" with l[0]=0
def _np_softmax(l):
  exp_l = np.exp(np.concatenate(([0.], l)))
  return exp_l / exp_l.sum()

class Result(np.ndarray): # https://stackoverflow.com/a/67510022/20580159
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

class GenericMethod:
  def __init__(self, loss, solver="trust-exact", seed=None):
    self.loss = loss
    self.solver = solver
    self.seed = seed
  def solve(self, q, M):
    loss_dict = losses.instantiate_loss(self.loss, q, M)
    rng = np.random.RandomState(self.seed)
    opt = optimize.minimize(
      loss_dict["fun"],
      rng.rand(loss_dict["n_classes"]-1), # l_0
      jac = loss_dict["jac"],
      hess = loss_dict["hess"],
      method = self.solver,
    )
    return Result(_np_softmax(opt.x), opt.nit, opt.message)
