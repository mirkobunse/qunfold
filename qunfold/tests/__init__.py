import numpy as np
from qunfold import CombinedLoss, GenericMethod, LeastSquaresLoss
from unittest import TestCase

RNG = np.random.RandomState(876) # make tests reproducible

def make_problem(n_features=None, n_classes=None):
  if n_classes is None:
    n_classes = RNG.randint(2, 5)
  if n_features is None:
    n_features = RNG.randint(n_classes, 12)
  M = .1 * RNG.rand(n_features, n_classes) + np.eye(n_features, n_classes)
  p_true = RNG.rand(n_classes)
  for i in range(n_classes):
    M[i,:] /= np.sum(M[i,:])
  p_true /= np.sum(p_true)
  q = np.matmul(M, p_true)
  return q, M, p_true

class TestLeastSquaresLoss(TestCase):
  def test_LeastSquaresLoss(self):
    for _ in range(10):
      q, M, p_true = make_problem()
      m = GenericMethod(LeastSquaresLoss())
      p_est = m.solve(q, M)
      print(
        f"LSq: p_est = {p_est}",
        f"    p_true = {p_true}",
        f"    {p_est.nit:2d} it.; {p_est.message}",
        sep = "\n",
        end = "\n"*2
      )
      # self.assertTrue(...)
