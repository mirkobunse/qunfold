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
      n_classes = len(p_trn)
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
      p_kmme = qunfold.KMM('energy').fit(X_trn, y_trn).predict(X_tst)
      p_kmmg = qunfold.KMM('gaussian').fit(X_trn, y_trn).predict(X_tst)
      p_kmml = qunfold.KMM('laplacian').fit(X_trn, y_trn).predict(X_tst)
      p_rff = qunfold.KMM('rff').fit(X_trn, y_trn).predict(X_tst)
      p_kdemc = qunfold.KDEyHD(RandomForestClassifier(oob_score=True), bandwidth=0.1).fit(X_trn, y_trn).predict(X_tst)
      p_kdecs = qunfold.KDEyCS(RandomForestClassifier(oob_score=True), bandwidth=0.1).fit(X_trn, y_trn).predict(X_tst)
      p_kdeml = qunfold.KDEyML(RandomForestClassifier(oob_score=True), bandwidth=0.1).fit(X_trn, y_trn).predict(X_tst)
      p_custom = qunfold.GenericMethod( # a custom method
        qunfold.LeastSquaresLoss(),
        qunfold.HistogramTransformer(3)
      ).fit(X_trn, y_trn, n_classes).predict(X_tst)
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
        f"     p_kmme = {p_kmme}",
        f"             {p_kmme.nit} it.; {p_kmme.message}",
        f"     p_kmmg = {p_kmmg}",
        f"             {p_kmmg.nit} it.; {p_kmmg.message}",
        f"     p_kmml = {p_kmml}",
        f"             {p_kmml.nit} it.; {p_kmml.message}",
        f"     p_rff = {p_rff}",
        f"             {p_rff.nit} it.; {p_rff.message}",
        f"     p_kdemc = {p_kdemc}",
        f"             {p_kdemc.nit} it.; {p_kdemc.message}",
        f"     p_kdecs = {p_kdecs}",
        f"             {p_kdecs.nit} it.; {p_kdecs.message}",
        f"     p_kdeml = {p_kdeml}",
        f"             {p_kdeml.nit} it.; {p_kdeml.message}",
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
      n_classes = len(p_trn)
      X_trn, y_trn = generate_data(M, p_trn)
      p_tst = RNG.permutation(p_trn)
      X_tst, y_tst = generate_data(M, p_tst)
      lr = CVClassifier(
        LogisticRegression(),
        n_estimators = 10,
        random_state = RNG.randint(np.iinfo("uint16").max),
      )
      p_acc = qunfold.ACC(lr).fit(X_trn, y_trn, n_classes).predict(X_tst)
      p_pacc = qunfold.PACC(lr).fit(X_trn, y_trn, n_classes).predict(X_tst)
      p_hdy = qunfold.HDy(lr, 3).fit(X_trn, y_trn, n_classes).predict(X_tst)
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
      self.assertEqual( # check that get_params returns the correct settings
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
      self.assertEqual( # check that best parameters are actually used
        quapy_method.best_params_["transformer__classifier__estimator__C"],
        quapy_method.best_model_.generic_method.transformer.classifier.estimator.C
      )

class TestDistanceTransformer(TestCase):
  def test_transformer(self):
    for _ in range(10):
      q, M, p_trn = make_problem()
      n_classes = len(p_trn)
      X_trn, y_trn = generate_data(M, p_trn)
      # p_tst = RNG.permutation(p_trn)
      # X_tst, y_tst = generate_data(M, p_tst)
      m = qunfold.GenericMethod(None, qunfold.DistanceTransformer())
      m.fit(X_trn, y_trn, n_classes)
      M_est = m.M
      M_true = np.zeros_like(M_est)
      for i in range(len(p_trn)):
        for j in range(len(p_trn)):
          M_true[i, j] = cdist(X_trn[y_trn==j], X_trn[y_trn==i]).mean()
      np.testing.assert_allclose(M_est, M_true)

class TestHistogramTransformer(TestCase):
  def test_transformer(self):
    X = np.load("qunfold/tests/HDx_X.npy")
    y = RNG.choice(5, size=X.shape[0]) # the HistogramTransformer ignores labels
    fX = np.load("qunfold/tests/HDx_fX.npy") # ground-truth by QUnfold.jl
    f = qunfold.HistogramTransformer(10, unit_scale=False)
    self.assertTrue(np.all(f.fit_transform(X, y, average=False)[0] == fX))
    self.assertTrue(np.all(f.transform(X, average=False) == fX))
    self.assertTrue(np.all(f.transform(X, average=True) == fX.mean(axis=0)))

    # test unit_scale=True, the new default
    self.assertTrue(np.all(f.transform(X, average=False).sum(axis=1) == X.shape[1]))
    f2 = qunfold.HistogramTransformer(10)
    self.assertTrue(np.allclose(f2.fit_transform(X, y, average=False)[0].sum(axis=1), 1))

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
      _, M, p_true = make_problem(10, 4)
      X_trn, y_trn = generate_data(M, p_true)
      y_trn -= y_trn.min() # map to zero-based labels
      n_bins = RNG.randint(2, 11)

      m_hl = HDx(n_bins).fit(X_trn, y_trn)
      M_hl = m_hl.M
      q_hl = m_hl.transformer.transform(X_trn, average=False).mean(axis=0)
      F_hl = jnp.sum(q_hl) # the number of features

      # draw a random p uniformly from the unit simplex, so the distance isn't just 0
      p_tst = RNG.dirichlet(np.ones(len(m_hl.p_trn)))

      new_loss = HellingerSurrogateLoss()._instantiate(q_hl, M_hl)
      old_loss = self._old_instantiate(q_hl, M_hl, n_bins=n_bins)

      # make sure both loss functions return roughly 0 for the true distribution
      self.assertAlmostEqual(F_hl + new_loss(m_hl.p_trn), 0, places=5)
      self.assertAlmostEqual(0.5 * old_loss(m_hl.p_trn), 0, places=5)

      # make sure the functions return roughly the same for other distributions
      # check for 6 decimal place accuracy
      self.assertAlmostEqual(F_hl + new_loss(p_tst),
                             0.5 * old_loss(p_tst),
                             places=5)
