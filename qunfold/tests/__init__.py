import numpy as np
from qunfold import losses
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
      for softmax in [True, False]:
        name = "softm." if softmax else "orig."
        result = losses.Result(losses.LeastSquaresLoss(q, M, softmax=softmax))
        print(f"LSq {name+':':<7} {result.iterations:2d} it. {result.p_est}")
        # self.assertTrue(...)
      print(" "*10 + f"p_true = {p_true}\n")
