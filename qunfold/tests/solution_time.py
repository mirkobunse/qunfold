import argparse
import itertools
import jax.numpy as jnp
import numpy as np
import pandas as pd
import quapy as qp
import qunfold
import matplotlib.pyplot as plt
from fact.io import read_data
from functools import partial
from multiprocessing import Pool
from qunfold import ClassTransformer, FunctionLoss, GenericMethod, LeastSquaresLoss
from qunfold.ensembles import EnsembleTransformer
from qunfold.tests import RNG, make_problem, generate_data
from sklearn.ensemble import RandomForestClassifier
from time import time

def load_fact(path="gamma_simulations_facttools_dl2.hdf5", n_samples=None):
    """This function loads the FACT data from a file."""
    df = read_data(path, key="events", last=n_samples) # read HDF5 file
    df["area"] = df["width"] * df["length"] * np.pi # generate more features
    df["log_size"] = np.log(df["size"])
    df["size_area"] = df["size"] / df["area"]
    df["cog_r"] = np.sqrt(df["cog_x"]**2 + df["cog_y"]**2)
    X = df[[ # select features
        "size",
        "width",
        "length",
        "skewness_trans",
        "skewness_long",
        "concentration_cog",
        "concentration_core",
        "concentration_one_pixel",
        "concentration_two_pixel",
        "leakage1",
        "leakage2",
        "num_islands",
        "num_pixel_in_shower",
        "photoncharge_shower_mean",
        "photoncharge_shower_variance",
        "photoncharge_shower_max",
        "area",
        "log_size",
        "size_area",
        "cog_r",
    ]].to_numpy()
    y = np.digitize( # bin the energy to obtain 12 classes (0, 1, ..., 11)
        np.log10(df["corsika_event_header_total_energy"]),
        np.linspace(2.4, 4.2, 13)[1:-1]
    )
    y = y[~np.isnan(X).any(1)] # replace NaNs and Infs
    X = X[~np.isnan(X).any(1)].astype(np.float32)
    X[np.isposinf(X)] = np.finfo('float32').max
    X[np.isneginf(X)] = np.finfo('float32').min
    return X, y

def scaled_lsq(p, q, M, N=None, scaling=1.):
  v = q - jnp.dot(M, p)
  return scaling * jnp.dot(v, v)

def trial(trial_args, trn_data, val_data, n_trials, seed):
  n_estimators, strategy = trial_args
  val_gen = qp.protocol.UPP( # instantiate the APP protocol
    val_data,
    sample_size = 1000,
    repeats = n_trials+1,
    random_state = seed
  )

  # configure and test the transformer; also set up the loss
  def create_classifier():
    return RandomForestClassifier(1, oob_score=True, random_state=seed)
  if strategy in [ "dup", "dup_scl" ]: # an EnsembleTransformer that duplicates a single member
    transformer = EnsembleTransformer(
      ClassTransformer(create_classifier()),
      n_estimators,
    )
    M = transformer.fit_transform(*trn_data.Xy, trn_data.n_classes)
    if n_estimators > 1:
      np.testing.assert_equal(M[0:12], M[12:24]) # check that it's actually duplicating
      np.testing.assert_equal(M[0:12], M[24:36])
  elif strategy in [ "app", "app_scl" ]: # a transformer with APP-randomized members
    transformer = EnsembleTransformer(
      ClassTransformer(create_classifier()),
      n_estimators,
      random_state = seed,
      training_strategy = "app"
    )
    M = transformer.fit_transform(*trn_data.Xy, trn_data.n_classes)
    if n_estimators > 1:
      assert np.any(M[0:12] != M[12:24])
      assert np.any(M[0:12] != M[24:36])
  if strategy in [ "dup", "app" ]:
    loss = LeastSquaresLoss()
  elif strategy in [ "dup_scl", "app_scl" ]:
    loss = FunctionLoss(partial(scaled_lsq, scaling=1 / n_estimators**2))

  # evaluate the transformer on validation samples
  generic_method = GenericMethod(
    loss,
    transformer,
    seed = seed,
    solver_options = {"gtol": 0, "maxiter": 1000}
  )
  results = []
  for i_trial, (X_tst, p_tst) in enumerate(val_gen()):
    q = transformer.transform(X_tst)

    # test the transformer again on q
    if n_estimators > 1:
      if strategy == "dup": # check that it's actually duplicating
        np.testing.assert_equal(q[0:12], q[12:24])
        np.testing.assert_equal(q[0:12], q[24:36])
        p_rnd = np.random.default_rng(i_trial).dirichlet(np.ones(12))
        np.testing.assert_approx_equal(
          qunfold.losses._lsq(p_rnd, q[0:12] * n_estimators, M[0:12] * n_estimators),
          scaled_lsq(p_rnd, q, M, scaling=n_estimators),
          significant = 4,
        )
      elif strategy == "app": # check that members differ
        assert np.any(q[0:12] != q[12:24])
        assert np.any(q[0:12] != q[24:36])

    # measure the time needed
    t_0 = time()
    generic_method.solve(q, M)
    t_solve = time() - t_0

    # store the results
    if i_trial > 0: # skip the first trial because its time might be misleading
      print(f"[{strategy:>7s}|n={n_estimators:<3d}|i={i_trial:>2d}/{n_trials}] t_solve = {t_solve:.4f}")
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
    df[("app", "25%")],
    df[("app", "75%")],
    color = "tab:orange",
    alpha = .1,
  )
  plt.fill_between(
    df.index,
    df[("dup_scl", "25%")],
    df[("dup_scl", "75%")],
    color = "tab:green",
    alpha = .1,
  )
  plt.fill_between(
    df.index,
    df[("app_scl", "25%")],
    df[("app_scl", "75%")],
    color = "tab:red",
    alpha = .1,
  )
  plt.plot(df.index, df[("dup", "mean")], color="tab:blue", label="dup")
  plt.plot(df.index, df[("app", "mean")], color="tab:orange", label="app")
  plt.plot(df.index, df[("dup_scl", "mean")], color="tab:green", label="dup_scl")
  plt.plot(df.index, df[("app_scl", "mean")], color="tab:red", label="app_scl")
  plt.xscale("log")
  plt.xlabel("n_estimators")
  plt.ylabel("t_solve")
  plt.legend()
  plt.show()

def main(
    strategies = [ "dup", "app", "dup_scl", "app_scl" ],
    n_estimators = [1, 3, 10, 31, 100, 316, 1000][::-1],
    n_trials = 100,
    n_jobs = 1,
    seed = 25,
    is_test_run = False
  ):
  if is_test_run:
    print("WARNING: This is a test run; results are unreliable")
    n_estimators = [ 3, 1 ]
    n_trials = 3

  # load the FACT data
  rem_data = qp.data.LabelledCollection(*load_fact())
  trn_data, rem_data = rem_data.split_stratified(train_prop=10000, random_state=seed)
  val_data, tst_data = rem_data.split_stratified(train_prop=.5, random_state=seed)
  print(f"Split into {len(trn_data)} training and {len(val_data)} validation items")

  # compute in parallel
  configured_trial = partial(trial, trn_data=trn_data, val_data=val_data, n_trials=n_trials, seed=seed)
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
  parser.add_argument('--seed', type=int, default=25, metavar='N',
                      help='random number generator seed (default: 25)')
  parser.add_argument("--is_test_run", action="store_true")
  args = parser.parse_args()
  main(
    # args.output_path,
    n_jobs = args.n_jobs,
    seed = args.seed,
    is_test_run = args.is_test_run,
  )
