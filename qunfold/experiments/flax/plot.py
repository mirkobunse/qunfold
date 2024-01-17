import argparse
import numpy as np
import pandas as pd
import quapy as qp
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

_CONFIG_COLS_MAIN = ["n_features", "lr_init", "lr_shrinkage", "batch_size"]
_CONFIG_COLS_PACC = ["n_features", "learning_rate", "batch_size"]

def baseline(n_val=64, use_precomputed_value=True): # baseline performance: SLD, the winner @ LeQua2022
  if use_precomputed_value: # must not be used when n_val changes
    return 0.012197593994264684
  print("Training and evaluating the baseline...")
  trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")
  X_trn, y_trn = trn_data.Xy
  val_gen.true_prevs.df = val_gen.true_prevs.df[:n_val] # use only n_jobs validation samples
  baseline_mae = qp.evaluation.evaluate( # average of all predictions
    qp.method.aggregative.EMQ(LogisticRegression(C=0.01)).fit(trn_data),
    protocol = val_gen,
    error_metric = "mae"
  )
  print(f"Baseline performance: SLD(LR(C=0.01)) = {baseline_mae}")
  return baseline_mae

def read_best(results_path, config_cols, keep_dominated=False):
  df = pd.read_csv(results_path, index_col=0).set_index(config_cols) # read the results
  n_configurations = len(df.groupby(config_cols))
  print(f"Read the results of {n_configurations} configurations from {results_path}")

  # find out which learning curves are dominated by others
  if keep_dominated:
    return df
  def is_dominated(sdf1):
    for _, sdf2 in df.groupby(config_cols):
      if np.all(sdf2["mae"].to_numpy()[10:] < sdf1["mae"].to_numpy()[10:]):
        return True # np.ones(len(sdf1), dtype=bool)
    return False # np.zeros(len(sdf1), dtype=bool)
  df["is_dominated"] = df.groupby(config_cols).apply(is_dominated)
  df = df[np.logical_not(df["is_dominated"])]
  n_remaining = len(df.groupby(config_cols))
  print(f"Removed {n_configurations - n_remaining} configurations for being dominated")
  return df

def main(main_path, pacc_path):
  df_main = read_best(main_path, _CONFIG_COLS_MAIN)
  df_pacc = read_best(pacc_path, _CONFIG_COLS_PACC)
  baseline_mae = baseline()

  fig, ax = plt.subplots() # figsize=(5, 2.7)
  for name, sdf in df_main.groupby(_CONFIG_COLS_MAIN): # plot main results
    ax.plot(sdf["batch_index"], sdf["mae"], label=f"$\phi$ {name}")
  for name, sdf in df_pacc.groupby(_CONFIG_COLS_PACC): # plot PACC results
    ax.plot(sdf["batch_index"] * 10, sdf["mae"], linestyle="dashed", label=f"PACC {name}")
  ax.axhline(baseline_mae, linestyle="dotted", color="gray", label="SLD(LR) (baseline)")
  ax.set_xlabel("n_batches")
  ax.set_ylabel("MAE")
  ax.legend(fontsize='xx-small')
  plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('main_path', type=str, help='path of an input *.csv file')
  parser.add_argument('pacc_path', type=str, help='path of an input *.csv file')
  args = parser.parse_args()
  main(
      args.main_path,
      args.pacc_path,
  )
