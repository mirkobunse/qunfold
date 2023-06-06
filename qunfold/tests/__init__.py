import numpy as np
import qunfold
import time
from quapy.data import LabelledCollection
from quapy.model_selection import GridSearchQ
from quapy.protocol import AbstractProtocol
from qunfold.quapy import QuaPyWrapper
from qunfold.sklearn import CVClassifier
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

def generate_data(M, p, n_samples=1000):
  n_features, n_classes = M.shape
  y = RNG.choice(np.arange(n_classes), p=p/p.sum(), size=n_samples)
  X = np.zeros((n_samples, n_features))
  for c in range(n_classes):
    X[y==c,:] = RNG.choice(
      np.arange(n_classes),
      p = M[c,:] / M[c,:].sum(),
      size = (np.sum(y==c), n_features)
    )
  X += RNG.rand(*X.shape) * .1
  return X, y

class TestMethods(TestCase):
  def test_methods(self):
    start = time.time()
    for _ in range(10):
      q, M, p_trn = make_problem()
      X_trn, y_trn = generate_data(M, p_trn)
      p_tst = RNG.permutation(p_trn)
      X_tst, y_tst = generate_data(M, p_tst)
      rf = RandomForestClassifier(
        oob_score = True,
        random_state = RNG.randint(np.iinfo("uint16").max),
      )
      p_acc = qunfold.ACC(rf).fit(X_trn, y_trn).predict(X_tst)
      p_pacc = qunfold.PACC(rf).fit(X_trn, y_trn).predict(X_tst)
      p_run = qunfold.RUN(qunfold.transformers.ClassTransformer(rf), tau=1e6).fit(X_trn, y_trn).predict(X_tst)
      p_hdx = qunfold.HDx(3).fit(X_trn, y_trn).predict(X_tst)
      p_hdy = qunfold.HDy(rf, 3).fit(X_trn, y_trn).predict(X_tst)
      p_edx = qunfold.EDx().fit(X_trn, y_trn).predict(X_tst)
      p_edy = qunfold.EDy(rf).fit(X_trn, y_trn).predict(X_tst)
      p_custom = qunfold.GenericMethod( # a custom method
        qunfold.LeastSquaresLoss(),
        qunfold.HistogramTransformer(3)
      ).fit(X_trn, y_trn).predict(X_tst)
      print(
        f"LSq: p_acc = {p_acc}",
        f"             {p_acc.nit} it.; {p_acc.message}",
        f"    p_pacc = {p_pacc}",
        f"             {p_pacc.nit} it.; {p_pacc.message}",
        f"     p_run = {p_run}",
        f"             {p_run.nit} it.; {p_run.message}",
        f"     p_hdx = {p_hdx}",
        f"             {p_hdx.nit} it.; {p_hdx.message}",
        f"     p_hdy = {p_hdy}",
        f"             {p_hdy.nit} it.; {p_hdy.message}",
        f"     p_edx = {p_edx}",
        f"             {p_edx.nit} it.; {p_edx.message}",
        f"     p_edy = {p_edy}",
        f"             {p_edy.nit} it.; {p_edy.message}",
        f"     p_custom = {p_custom}",
        f"             {p_custom.nit} it.; {p_custom.message}",
        f"     p_tst = {p_tst}",
        sep = "\n",
        end = "\n"*2
      )
      # self.assertTrue(...)
    print(f"Spent {time.time() - start}s")

class TestCVClassifier(TestCase):
  def test_methods(self):
    start = time.time()
    for _ in range(10):
      q, M, p_trn = make_problem()
      X_trn, y_trn = generate_data(M, p_trn)
      p_tst = RNG.permutation(p_trn)
      X_tst, y_tst = generate_data(M, p_tst)
      lr = CVClassifier(
        LogisticRegression(),
        n_estimators = 10,
        random_state = RNG.randint(np.iinfo("uint16").max),
      )
      p_acc = qunfold.ACC(lr).fit(X_trn, y_trn).predict(X_tst)
      p_pacc = qunfold.PACC(lr).fit(X_trn, y_trn).predict(X_tst)
      p_hdy = qunfold.HDy(lr, 3).fit(X_trn, y_trn).predict(X_tst)
      print(
        f"CVC: p_acc = {p_acc}",
        f"             {p_acc.nit} it.; {p_acc.message}",
        f"    p_pacc = {p_pacc}",
        f"             {p_pacc.nit} it.; {p_pacc.message}",
        f"     p_hdy = {p_hdy}",
        f"             {p_hdy.nit} it.; {p_hdy.message}",
        f"     p_tst = {p_tst}",
        sep = "\n",
        end = "\n"*2
      )
      # self.assertTrue(...)
    print(f"Spent {time.time() - start}s")

class SingleSampleProtocol(AbstractProtocol):
    def __init__(self, X, p):
      self.X = X
      self.p = p
    def __call__(self):
      yield self.X, self.p

class TestQuaPyWrapper(TestCase):
  def test_methods(self):
    for _ in range(10):
      q, M, p_trn = make_problem()
      X_trn, y_trn = generate_data(M, p_trn)
      p_tst = RNG.permutation(p_trn)
      X_tst, y_tst = generate_data(M, p_tst)
      lr = CVClassifier(
        LogisticRegression(C = 1e-2), # some value outside of the param_grid
        n_estimators = 10,
        random_state = RNG.randint(np.iinfo("uint16").max),
      )
      p_acc = QuaPyWrapper(qunfold.ACC(lr))
      self.assertEquals( # check that get_params returns the correct settings
        p_acc.get_params(deep=True)["transformer__classifier__estimator__C"],
        1e-2
      )
      quapy_method = GridSearchQ(
        model = p_acc,
        param_grid = {
          "transformer__classifier__estimator__C": [1e-1, 1e0, 1e1, 1e2],
        },
        protocol = SingleSampleProtocol(X_tst, p_tst),
        error = "mae",
        refit = False,
        verbose = True,
      ).fit(LabelledCollection(X_trn, y_trn))
      self.assertEquals( # check that best parameters are actually used
        quapy_method.best_params_["transformer__classifier__estimator__C"],
        quapy_method.best_model_.generic_method.transformer.classifier.estimator.C
      )

class TestDistanceTransformer(TestCase):
  def test_transformer(self):
    for _ in range(10):
      q, M, p_trn = make_problem()
      X_trn, y_trn = generate_data(M, p_trn)
      # p_tst = RNG.permutation(p_trn)
      # X_tst, y_tst = generate_data(M, p_tst)
      m = qunfold.GenericMethod(None, qunfold.DistanceTransformer())
      m.fit(X_trn, y_trn)
      M_est = m.M
      M_true = np.zeros_like(M_est)
      for i in range(len(p_trn)):
        for j in range(len(p_trn)):
          M_true[i, j] = cdist(X_trn[y_trn==j], X_trn[y_trn==i]).mean()
      np.testing.assert_allclose(M_est, M_true)

class TestHistogramTransformer(TestCase):
  def test_transformer(self):
    X = np.load("qunfold/tests/HDx_X.npy")
    y = np.random.choice(5, size=X.shape[0]) # the HistogramTransformer ignores labels
    fX = np.load("qunfold/tests/HDx_fX.npy") # ground-truth by QUnfold.jl
    f = qunfold.HistogramTransformer(10, unit_scale=False)
    self.assertTrue(np.all(f.fit_transform(X, y)[0] == fX))
    self.assertTrue(np.all(f.transform(X) == fX))

    # test unit_scale=True, the new default
    self.assertTrue(np.all(f.transform(X).sum(axis=1) == X.shape[1]))
    f2 = qunfold.HistogramTransformer(10)
    self.assertTrue(np.allclose(f2.fit_transform(X, y)[0].sum(axis=1), 1))
