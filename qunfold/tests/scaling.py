import numpy as np
import quapy as qp
import qunfold
from functools import partial
from qunfold import ClassTransformer, FunctionLoss, GenericMethod, LeastSquaresLoss
from qunfold.tests.solution_time import load_fact, scaled_lsq
from qunfold.ensembles import EnsembleTransformer
from sklearn.ensemble import RandomForestClassifier

def main(
    n_estimators = 5,
    n_trials = 10,
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

  # configure two methods
  print(f"Fitting the two methods...")
  m_dup = GenericMethod( # method with a scaled loss to counteract duplication
    FunctionLoss(partial(scaled_lsq, scaling=1 / n_estimators)),
    EnsembleTransformer( # an EnsembleTransformer that duplicates a single member
      ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=seed)),
      n_estimators,
    ),
    seed = seed,
    solver_options = {"gtol": 0, "maxiter": 1000}
  ).fit(*trn_data.Xy)
  m_reg = GenericMethod( # method with a regular loss and a regular transformer
    FunctionLoss(partial(scaled_lsq, scaling=1)), # LeastSquaresLoss(),
    ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=seed)),
    seed = seed,
    solver_options = {"gtol": 0, "maxiter": 1000}
  ).fit(*trn_data.Xy)
  M_dup = m_dup.M
  M_reg = m_reg.M
  for i_trial, (X_tst, p_tst) in enumerate(val_gen()):
    print(f"Testing with sample {i_trial+1}/{n_trials}...")
    x0 = qunfold.methods._rand_x0(np.random.RandomState(seed), 12)
    jac_dup = qunfold.losses.instantiate_loss(
      m_dup.loss,
      m_dup.transformer.transform(X_tst), # q
      M_dup,
      X_tst.shape[0] # N
    )["jac"](x0)
    jac_reg = qunfold.losses.instantiate_loss(
      m_reg.loss,
      m_reg.transformer.transform(X_tst), # q
      M_reg,
      X_tst.shape[0] # N
    )["jac"](x0)
    print(jac_dup / jac_reg)
    p_dup = m_dup.predict(X_tst)
    p_reg = m_reg.predict(X_tst)
    p_rep = m_reg.predict(X_tst) # repeat unscaled solution
    print(f"...p_dup.nit = {p_dup.nit}, p_reg.nit = {p_reg.nit}")
    np.testing.assert_equal(p_reg, p_rep) # no randomness
    np.testing.assert_equal(p_reg, p_dup) # the actual test

if __name__ == '__main__':
  main()
