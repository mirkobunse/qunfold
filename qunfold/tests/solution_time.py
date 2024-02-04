import argparse
import itertools
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

def trial(trial_args, trn_data, val_gen):
  n_estimators, strategy = trial_args

  # configure and test the transformer; also set up the loss
  if strategy in [ "dup", "dup_nrm" ]: # an EnsembleTransformer that duplicates a single member
    transformer = EnsembleTransformer(
      ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=0)),
      n_estimators,
    )
    M = transformer.fit_transform(*trn_data.Xy, trn_data.n_classes)
    if n_estimators > 1:
      np.testing.assert_equal(M[0:28], M[28:56]) # check that it's actually duplicating
      np.testing.assert_equal(M[0:28], M[56:84])
    if strategy == "dup":
      loss = LeastSquaresLoss()
    elif strategy == "dup_nrm":
      loss = FunctionLoss(partial(scaled_lsq, scaling=1/n_estimators))

  elif strategy == "scl": # no duplication, this is actually just a single transformer
    transformer = ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=0))
    M = transformer.fit_transform(*trn_data.Xy, trn_data.n_classes)
    if n_estimators == 1:
      np.testing.assert_equal( # a one-member ensemble must match a single transformer
        M,
        EnsembleTransformer(
          ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=0)),
          1,
        ).fit_transform(*trn_data.Xy, trn_data.n_classes)
      )
    loss = FunctionLoss(partial(scaled_lsq, scaling=n_estimators))

  elif strategy == "app_nrm": # a transformer with APP-randomized members
    transformer = EnsembleTransformer(
      ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=0)),
      n_estimators,
      random_state = 0,
      training_strategy = "app"
    )
    M = transformer.fit_transform(*trn_data.Xy, trn_data.n_classes)
    if n_estimators > 1:
      assert np.any(M[0:28] != M[28:56])
      assert np.any(M[0:28] != M[56:84])
    loss = FunctionLoss(partial(scaled_lsq, scaling=1/n_estimators))

  # evaluate the transformer on validation samples
  generic_method = GenericMethod(loss, transformer, seed=0)
  results = []
  for i_trial, (X_tst, p_tst) in enumerate(val_gen()):
    q = transformer.transform(X_tst)

    # test the transformer again on q
    if n_estimators > 1:
      if strategy == "dup": # check that it's actually duplicating
        np.testing.assert_equal(q[0:28], q[28:56])
        np.testing.assert_equal(q[0:28], q[56:84])
      elif strategy == "app_nrm": # check that members differ
        assert np.any(q[0:28] != q[28:56])
        assert np.any(q[0:28] != q[56:84])

    # measure the time needed
    t_0 = time()
    generic_method.solve(q, M)
    t_solve = time() - t_0

    # store the results
    if i_trial > 0: # skip the first trial because its time might be misleading
      print(f"[{strategy:>7s}|n={n_estimators:<3d}|i={i_trial:>2d}/{len(val_gen.true_prevs.df)-1}] t_solve = {t_solve:.4f}")
      results.append({
        "n_estimators": n_estimators,
        "strategy": strategy,
        "trial": i_trial,
        "t_solve": t_solve,
      })
  return results

def plot(df):
  df = df \
    .pivot(index=["n_estimators", "trial"], columns="strategy", values="t_solve") \
    .groupby("n_estimators") \
    .describe()
  plt.fill_between(
    df.index,
    df[("dup", "25%")],
    df[("dup", "75%")],
    color = "tab:blue",
    alpha = .1,
  )
  plt.fill_between(
    df.index,
    df[("dup_nrm", "25%")],
    df[("dup_nrm", "75%")],
    color = "tab:orange",
    alpha = .1,
  )
  plt.fill_between(
    df.index,
    df[("scl", "25%")],
    df[("scl", "75%")],
    color = "tab:green",
    alpha = .1,
  )
  plt.fill_between(
    df.index,
    df[("app_nrm", "25%")],
    df[("app_nrm", "75%")],
    color = "tab:red",
    alpha = .1,
  )
  plt.plot(df.index, df[("dup", "mean")], color="tab:blue", label="dup")
  plt.plot(df.index, df[("dup_nrm", "mean")], color="tab:orange", label="dup_nrm")
  plt.plot(df.index, df[("scl", "mean")], color="tab:green", label="scl")
  plt.plot(df.index, df[("app_nrm", "mean")], color="tab:red", label="app_nrm")
  plt.xscale("log")
  plt.xlabel("n_estimators")
  plt.ylabel("t_solve")
  plt.legend()
  plt.show()

def main(
    strategies = [ "dup", "dup_nrm", "scl", "app_nrm" ],
    n_estimators = [3, 10, 31, 100, 316, 1000][::-1],
    n_trials = 100,
    n_jobs = 1,
  ):
  # load the LeQua data
  trn_data, val_gen, _ = qp.datasets.fetch_lequa2022(task="T1B")
  trn_data = trn_data.split_stratified(10000, random_state=0)[0] # limit to 10k training items
  val_gen.true_prevs.df = val_gen.true_prevs.df[:n_trials+1] # limit to n_trials samples

  # compute in parallel
  configured_trial = partial(trial, trn_data=trn_data, val_gen=val_gen)
  trials = itertools.product(n_estimators, strategies)
  results = []
  with Pool(n_jobs if n_jobs > 0 else None) as pool:
    for trial_results in pool.imap(configured_trial, trials):
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
