import argparse
import jax.numpy as jnp
import numpy as np
import pandas as pd
import quapy as qp
import qunfold
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool
from qunfold import ClassTransformer, GenericMethod, LeastSquaresLoss
from qunfold.ensembles import EnsembleTransformer
from qunfold.tests import RNG, make_problem, generate_data
from sklearn.ensemble import RandomForestClassifier
from time import time

def trial(n_estimators, trn_data, val_gen):

  # train an EnsembleTransformer that duplicates a single member
  duplicate_transformer = EnsembleTransformer(
    ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=0)),
    n_estimators
  )
  M_dup = duplicate_transformer.fit_transform(*trn_data.Xy, trn_data.n_classes)
  if n_estimators > 1:
    np.testing.assert_equal(M_dup[0:28], M_dup[28:56]) # check that it's actually duplicating
    np.testing.assert_equal(M_dup[0:28], M_dup[56:84])

  # train a transformer with differently seeded members
  randomized_transformer = EnsembleTransformer([
    ClassTransformer(RandomForestClassifier(1, oob_score=True, random_state=i))
    for i in range(n_estimators)
  ])
  M_rnd = randomized_transformer.fit_transform(*trn_data.Xy, trn_data.n_classes)
  if n_estimators > 1:
    assert np.any(M_rnd[0:28] != M_rnd[28:56])
    assert np.any(M_rnd[0:28] != M_rnd[56:84])

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

    # measure time to solve q=Mp with the randomized_transformer
    q = randomized_transformer.transform(X_tst)
    if n_estimators > 1:
      assert np.any(q[0:28] != q[28:56])
      assert np.any(q[0:28] != q[56:84])
    generic_method = GenericMethod(LeastSquaresLoss(), randomized_transformer, seed=0)
    t_0 = time()
    generic_method.solve(q, M_rnd)
    t_rnd = time() - t_0

    # measure time to solve q=Mp with the app_transformer
    q = app_transformer.transform(X_tst)
    if n_estimators > 1:
      assert np.any(q[0:28] != q[28:56])
      assert np.any(q[0:28] != q[56:84])
    generic_method = GenericMethod(LeastSquaresLoss(), app_transformer, seed=0)
    t_0 = time()
    generic_method.solve(q, M_app)
    t_app = time() - t_0

    # store the results
    if i_trial > 0:
      print(
        f"[n={n_estimators} | {i_trial:02d}/{len(val_gen.true_prevs.df)-1}]",
        f"t_dup={t_dup:.3f} t_rnd={t_rnd:.3f} t_app={t_app:.3f}"
      )
      results.append({
        "n_estimators": n_estimators,
        "trial": i_trial,
        "t_dup": t_dup,
        "t_rnd": t_rnd,
        "t_app": t_app,
      })
  return results

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
  df = pd.DataFrame(results)
  # df.to_csv(output_path) # TODO

  # avg time needed
  avg_df = df.groupby("n_estimators").aggregate([np.mean, np.std])
  plt.fill_between(
    avg_df.index,
    avg_df[("t_dup", "mean")] - avg_df[("t_dup", "std")],
    avg_df[("t_dup", "mean")] + avg_df[("t_dup", "std")],
    color = "tab:blue",
    alpha = .2,
  )
  plt.fill_between(
    avg_df.index,
    avg_df[("t_rnd", "mean")] - avg_df[("t_rnd", "std")],
    avg_df[("t_rnd", "mean")] + avg_df[("t_rnd", "std")],
    color = "tab:orange",
    alpha = .2,
  )
  plt.fill_between(
    avg_df.index,
    avg_df[("t_app", "mean")] - avg_df[("t_app", "std")],
    avg_df[("t_app", "mean")] + avg_df[("t_app", "std")],
    color = "tab:green",
    alpha = .2,
  )
  plt.loglog(avg_df.index, avg_df[("t_dup", "mean")], color="tab:blue", label="t_dup")
  plt.loglog(avg_df.index, avg_df[("t_rnd", "mean")], color="tab:orange", label="t_rnd")
  plt.loglog(avg_df.index, avg_df[("t_app", "mean")], color="tab:green", label="t_app")
  plt.legend()
  plt.show()

  # TODO CD diagram?
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
