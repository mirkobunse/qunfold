import numpy as np
import jax.numpy as jnp
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
from qunfold.methods import HDx
from qunfold.losses import HellingerSurrogateLoss

RNG = np.random.RandomState(877) # make tests reproducible

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
    self.assertTrue(np.all(f.transform(X, average=True) == fX.mean(axis=0)))

    # test unit_scale=True, the new default
    self.assertTrue(np.all(f.transform(X).sum(axis=1) == X.shape[1]))
    f2 = qunfold.HistogramTransformer(10)
    self.assertTrue(np.allclose(f2.fit_transform(X, y)[0].sum(axis=1), 1))

class TestHellingerSurrogateLoss(TestCase):
  #old implementation of hellinger surrogate loss for comparison
  def old_hellinger_loss_function(self, p, q, M, indices):
    v = (jnp.sqrt(q) - jnp.sqrt(jnp.dot(M, p)))**2
    return jnp.sum(jnp.array([ jnp.sum(v[i]) for i in indices ]))

  # old instantiation of hellinger surrogate loss for comparison
  def _old_instantiate(self, q, M, n_bins, N=None):
    n_features = int(M.shape[0] / n_bins) # derive the number from M's shape
    indices = [ jnp.arange(i * n_bins, (i+1) * n_bins) for i in range(n_features) ]
    nonzero = jnp.any(M != 0, axis=1)
    M = M[nonzero,:]
    q = q[nonzero]
    if not jnp.all(nonzero):
      i = 0
      for j in range(len(indices)):
        indices_j = []
        for k in indices[j]:
          if nonzero[k]:
            indices_j.append(i)
            i += 1
        indices[j] = jnp.array(indices_j, dtype=int)
    return lambda p: self.old_hellinger_loss_function(p, q, M, indices)
  
  def test_loss(self):
    for _ in range(20):
      q_hl, M_hl, p_true_hl = make_problem(10, 4)
      X_tst_hl, y_tst_hl = generate_data(M_hl, p_true_hl)

      n_bins_hl = np.random.randint(2, 11)

      fitted_hl = HDx(n_bins_hl).fit(X_tst_hl, y_tst_hl)

      M_tst_hl = fitted_hl.M
      q_hl = fitted_hl.transformer.transform(X_tst_hl, average=False).mean(axis=0)

      # vary p so the distance isnt just 0
      p_tst_hl = p_true_hl + np.random.randint(1, 25, p_true_hl.shape[0])
      p_tst_hl /= p_tst_hl.sum()

      F_hl = jnp.sum(q_hl)

      new_loss = HellingerSurrogateLoss()._instantiate(q_hl, M_tst_hl)
      old_loss = self._old_instantiate(q_hl, M_tst_hl, n_bins=n_bins_hl)

      # make sure both loss functions return roughly 0 for the true distribution
      self.assertAlmostEqual(F_hl + new_loss(fitted_hl.p_trn), 0, places=5)
      self.assertAlmostEqual(0.5 * old_loss(fitted_hl.p_trn), 0, places=5)

      # make sure the functions return roughly the same for other distributions
      # check for 6 decimal place accuracy
      self.assertAlmostEquals(F_hl + new_loss(p_tst_hl),
                              0.5 * old_loss(p_tst_hl),
                              places=5) 