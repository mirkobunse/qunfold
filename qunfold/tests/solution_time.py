import argparse
import jax.numpy as jnp
import numpy as np
import pandas as pd
import quapy as qp
import qunfold
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool
from qunfold import ClassTransformer, FunctionLoss, GenericMethod, LeastSquaresLoss
from qunfold.ensembles import EnsembleTransformer
from qunfold.tests import RNG, make_problem, generate_data
from sklearn.ensemble import RandomForestClassifier
from time import time

def scaled_lsq(p, q, M, N=None, scaling=1.):
  v = q - jnp.dot(M, p)
  return scaling * jnp.dot(v, v)

def trial(n_estimators, trn_data, val_gen):

  # train an EnsembleTransformer that duplicates a single member
  duplicate_transformer = EnsembleTransformer(
    ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=0)),
    n_estimators
  )
  M_dup = duplicate_transformer.fit_transform(*trn_data.Xy, trn_data.n_classes)
  M_scl = M_dup[0:28] # only the first ensemble member
  if n_estimators > 1:
    np.testing.assert_equal(M_scl, M_dup[28:56]) # check that it's actually duplicating
    np.testing.assert_equal(M_scl, M_dup[56:84])

  # # train a transformer with differently seeded members
  # randomized_transformer = EnsembleTransformer([
  #   ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=i))
  #   for i in range(n_estimators)
  # ])
  # M_rnd = randomized_transformer.fit_transform(*trn_data.Xy, trn_data.n_classes)
  # if n_estimators > 1:
  #   assert np.any(M_rnd[0:28] != M_rnd[28:56])
  #   assert np.any(M_rnd[0:28] != M_rnd[56:84])

  # train a transformer with APP-randomized members
  app_transformer = EnsembleTransformer(
    ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=0)),
    n_estimators,
    training_strategy = "app"
  )
  M_app = app_transformer.fit_transform(*trn_data.Xy, trn_data.n_classes)
  if n_estimators > 1:
    assert np.any(M_app[0:28] != M_app[28:56])
    assert np.any(M_app[0:28] != M_app[56:84])

  # evaluate both transformers on validation samples
  results = []
  for i_trial, (X_tst, p_tst) in enumerate(val_gen()):

    # measure time to solve q=Mp with the duplicate_transformer
    q = duplicate_transformer.transform(X_tst)
    if n_estimators > 1:
      np.testing.assert_equal(q[0:28], q[28:56]) # check that it's actually duplicating
      np.testing.assert_equal(q[0:28], q[56:84])
    generic_method = GenericMethod(LeastSquaresLoss(), duplicate_transformer, seed=0)
    t_0 = time()
    generic_method.solve(q, M_dup)
    t_dup = time() - t_0

    # measure time with the duplicate_transformer again, this time with a normalized loss
    if n_estimators > 1:
      p_rnd = np.random.default_rng(i_trial).dirichlet(np.ones(28))
      np.testing.assert_approx_equal( # the scaled loss matches the original one
        scaled_lsq(p_rnd, q, M_dup, scaling=1/n_estimators),
        qunfold.losses._lsq(p_rnd, q[0:28], M_scl),
        significant = 3
      )
    generic_method = GenericMethod(
      FunctionLoss(partial(scaled_lsq, scaling=1/n_estimators)),
      duplicate_transformer,
      seed = 0
    )
    t_0 = time()
    generic_method.solve(q, M_dup)
    t_dup_nrm = time() - t_0

    # instead of normalizing, use only the first ensemble member but scale up the loss
    q = q[0:28]
    generic_method = GenericMethod(
      FunctionLoss(partial(scaled_lsq, scaling=n_estimators)),
      duplicate_transformer,
      seed = 0
    )
    t_0 = time()
    generic_method.solve(q, M_scl)
    t_scl = time() - t_0

    # # measure time to solve q=Mp with the randomized_transformer
    # q = randomized_transformer.transform(X_tst)
    # if n_estimators > 1:
    #   assert np.any(q[0:28] != q[28:56])
    #   assert np.any(q[0:28] != q[56:84])
    # generic_method = GenericMethod(LeastSquaresLoss(), randomized_transformer, seed=0)
    # t_0 = time()
    # generic_method.solve(q, M_rnd)
    # t_rnd = time() - t_0

    # measure time to solve q=Mp with the app_transformer (and a normalized loss)
    q = app_transformer.transform(X_tst)
    if n_estimators > 1:
      assert np.any(q[0:28] != q[28:56])
      assert np.any(q[0:28] != q[56:84])
    # generic_method = GenericMethod(LeastSquaresLoss(), app_transformer, seed=0)
    # t_0 = time()
    # generic_method.solve(q, M_app)
    # t_app = time() - t_0
    generic_method = GenericMethod(
      FunctionLoss(partial(scaled_lsq, scaling=1/n_estimators)),
      app_transformer,
      seed = 0
    )
    t_0 = time()
    generic_method.solve(q, M_app)
    t_app_nrm = time() - t_0

    # store the results
    if i_trial > 0: # skip the first trial because time might be misleading
      print(
        f"[n={n_estimators} | {i_trial:02d}/{len(val_gen.true_prevs.df)-1}]",
        f"t_dup={t_dup:.3f} t_dup_nrm={t_dup_nrm:.3f} t_scl={t_scl:.3f} t_app_nrm={t_app_nrm:.3f}"
      )
      results.append({
        "n_estimators": n_estimators,
        "trial": i_trial,
        "t_dup": t_dup,
        "t_dup_nrm": t_dup_nrm,
        "t_scl": t_scl,
        "t_app_nrm": t_app_nrm,
      })
  return results

def plot(df):
  df = df.groupby("n_estimators").describe()
  plt.fill_between(
    df.index,
    df[("t_dup", "25%")],
    df[("t_dup", "75%")],
    color = "tab:blue",
    alpha = .1,
  )
  plt.fill_between(
    df.index,
    df[("t_dup_nrm", "25%")],
    df[("t_dup_nrm", "75%")],
    color = "tab:orange",
    alpha = .1,
  )
  plt.fill_between(
    df.index,
    df[("t_scl", "25%")],
    df[("t_scl", "75%")],
    color = "tab:green",
    alpha = .1,
  )
  plt.fill_between(
    df.index,
    df[("t_app_nrm", "25%")],
    df[("t_app_nrm", "75%")],
    color = "tab:red",
    alpha = .1,
  )
  plt.plot(df.index, df[("t_dup", "mean")], color="tab:blue", label="t_dup")
  plt.plot(df.index, df[("t_dup_nrm", "mean")], color="tab:orange", label="t_dup_nrm")
  plt.plot(df.index, df[("t_scl", "mean")], color="tab:green", label="t_scl")
  plt.plot(df.index, df[("t_app_nrm", "mean")], color="tab:red", label="t_app_nrm")
  plt.xscale("log")
  plt.legend()
  plt.show()

def main(
    n_estimators = [3, 10, 31, 100, 316, 1000],
    n_trials = 100,
    n_jobs = 1,
  ):
  # load the LeQua data
  trn_data, val_gen, _ = qp.datasets.fetch_lequa2022(task="T1B")
  trn_data = trn_data.split_stratified(10000, random_state=0)[0] # limit to 10k training items
  val_gen.true_prevs.df = val_gen.true_prevs.df[:n_trials+1] # limit to n_trials samples

  # compute in parallel
  configured_trial = partial(trial, trn_data=trn_data, val_gen=val_gen)
  results = []
  with Pool(n_jobs if n_jobs > 0 else None) as pool:
    for trial_results in pool.imap(configured_trial, n_estimators):
      results.extend(trial_results)
  df = pd.DataFrame(results) # TODO write data? - df.to_csv(output_path)

  # plot and return the results
  plot(df) # TODO also add a CD diagram?
  return df

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument('output_path', type=str, help='path of an output *.csv file')
  parser.add_argument('--n_jobs', type=int, default=1, metavar='N',
                      help='number of concurrent jobs or 0 for all processors (default: 1)')
  # parser.add_argument('--seed', type=int, default=876, metavar='N',
  #                     help='random number generator seed (default: 876)')
  # parser.add_argument("--is_test_run", action="store_true")
  args = parser.parse_args()
  main(
    # args.output_path,
    n_jobs = args.n_jobs,
    # args.seed,
    # args.is_test_run,
  )
