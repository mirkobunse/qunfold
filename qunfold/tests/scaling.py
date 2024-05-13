import numpy as np
import quapy as qp
import qunfold
from functools import partial
from qunfold import ClassTransformer, FunctionLoss, GenericMethod, LeastSquaresLoss
from qunfold.tests.solution_time import load_fact, scaled_lsq
from qunfold.ensembles import EnsembleTransformer
from sklearn.ensemble import RandomForestClassifier

def main(
    optimization = "softmax",
    n_estimators = 5,
    n_trials = 5,
    seed = 25,
  ):
  print("Loading the FACT data...")
  rem_data = qp.data.LabelledCollection(*load_fact())
  trn_data, rem_data = rem_data.split_stratified(train_prop=10000, random_state=seed)
  val_data, tst_data = rem_data.split_stratified(train_prop=.5, random_state=seed)
  print(f"Split into {len(trn_data)} training and {len(val_data)} validation items")
  val_gen = qp.protocol.UPP( # instantiate the APP protocol
    val_data,
    sample_size = 1000,
    repeats = n_trials,
    random_state = seed
  )
  qp.environ["SAMPLE_SIZE"] = 1000

  # for each optimization, configure two methods
  for optimization in ["softmax", "constrained"]:
    print(f"Fitting the two methods for optimization={optimization}...")
    m_dup = GenericMethod( # method with a scaled loss to counteract duplication
      FunctionLoss(partial(scaled_lsq, scaling=1 / n_estimators)),
      EnsembleTransformer( # an EnsembleTransformer that duplicates a single member
        ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=seed)),
        n_estimators,
      ),
      optimization = optimization,
      seed = seed,
      solver = "trust-constr",
      solver_options = {"gtol": 0, "xtol": 0, "maxiter": 100}
    ).fit(*trn_data.Xy)
    m_reg = GenericMethod( # method with a regular loss and a regular transformer
      FunctionLoss(partial(scaled_lsq, scaling=1)), # LeastSquaresLoss(),
      ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=seed)),
      optimization = optimization,
      seed = seed,
      solver = "trust-constr",
      solver_options = {"gtol": 0, "xtol": 0, "maxiter": 100}
    ).fit(*trn_data.Xy)
    M_dup = m_dup.M
    M_reg = m_reg.M
    for i_trial, (X_tst, p_tst) in enumerate(val_gen()):
      print(f"Testing optimization={optimization} sample {i_trial+1}/{n_trials}...")
      x0 = qunfold.methods._rand_x0(
        np.random.RandomState(seed),
        n_classes = 12,
        use_logodds = optimization == "softmax",
      )
      jac_dup = qunfold.losses.instantiate_loss(
        m_dup.loss,
        m_dup.transformer.transform(X_tst), # q
        M_dup,
        X_tst.shape[0], # N
        use_logodds = optimization == "softmax",
      )["jac"](x0)
      jac_reg = qunfold.losses.instantiate_loss(
        m_reg.loss,
        m_reg.transformer.transform(X_tst), # q
        M_reg,
        X_tst.shape[0], # N
        use_logodds = optimization == "softmax",
      )["jac"](x0)
      # print(jac_dup / jac_reg)
      p_dup = m_dup.predict(X_tst)
      p_reg = m_reg.predict(X_tst)
      p_rep = m_reg.predict(X_tst) # repeat unscaled solution
      # print(f"p_dup.nit = {p_dup.nit}, p_reg.nit = {p_reg.nit}")
      np.testing.assert_equal(p_reg, p_rep) # no randomness
      print(
        f"RAE(p_reg)={qp.error.rae(p_reg, p_tst)}",
        f"RAE(p_dup)={qp.error.rae(p_dup, p_tst)}",
        f"impr(dup over reg)={qp.error.rae(p_reg, p_tst) - qp.error.rae(p_dup, p_tst)}",
      )
      # print(f"Max absolute difference: {np.max(np.abs(p_reg - p_dup))}")
      # print(f"Max relative difference: {np.max(np.abs(p_reg - p_dup) / p_reg)}")
      # np.testing.assert_equal(p_reg, p_dup) # the actual test

if __name__ == '__main__':
  main()
