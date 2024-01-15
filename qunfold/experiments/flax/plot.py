import argparse
import numpy as np
import pandas as pd
import quapy as qp
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

_CONFIG_COLS = ["n_features", "learning_rate", "batch_size"]

def baseline(n_val=64): # baseline performance: SLD, the winner @ LeQua2022
  trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")
  X_trn, y_trn = trn_data.Xy
  val_gen.true_prevs.df = val_gen.true_prevs.df[:n_val] # use only n_jobs validation samples
  return qp.evaluation.evaluate( # average of all predictions
    qp.method.aggregative.EMQ(LogisticRegression(C=0.01)).fit(trn_data),
    protocol = val_gen,
    error_metric = "mae"
  )

def main(results_path):
  df = pd.read_csv(results_path, index_col=0).set_index(_CONFIG_COLS) # read the results
  n_configurations = len(df.groupby(_CONFIG_COLS))
  print(f"Read the results of {n_configurations} configurations")

  # find out which learning curves are dominated by others
  def is_dominated(sdf1):
    for _, sdf2 in df.groupby(_CONFIG_COLS):
      if np.all(sdf2["mae"].to_numpy()[10:] < sdf1["mae"].to_numpy()[10:]):
        return True # np.ones(len(sdf1), dtype=bool)
    return False # np.zeros(len(sdf1), dtype=bool)
  df["is_dominated"] = df.groupby(_CONFIG_COLS).apply(is_dominated)
  df = df[np.logical_not(df["is_dominated"])]
  n_remaining = len(df.groupby(_CONFIG_COLS))
  print(f"Removed {n_configurations - n_remaining} configurations for being dominated")

  baseline_mae = baseline()
  print(f"Baseline performance: SLD(LR(C=0.01)) = {baseline_mae}")

  fig, ax = plt.subplots() # figsize=(5, 2.7)
  for name, sdf in df.groupby(_CONFIG_COLS):
    ax.plot(sdf["batch_index"], sdf["mae"], label=name)
  ax.axhline(baseline_mae, linestyle="dashed", color="gray", label="SLD(LR) (baseline)")
  ax.set_xlabel("n_batches")
  ax.set_ylabel("MAE")
  ax.legend()
  plt.show()
  
  # ax.plot(x, x, label='linear')  # Plot some data on the axes.
  # ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
  # ax.plot(x, x**3, label='cubic')  # ... and some more.
  # ax.set_xlabel('x label')  # Add an x-label to the axes.
  # ax.set_ylabel('y label')  # Add a y-label to the axes.
  # ax.set_title("Simple Plot")  # Add a title to the axes.
  # ax.legend()  # Add a legend.

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('results_path', type=str, help='path of an input *.csv file')
  args = parser.parse_args()
  main(
      args.results_path,
  )
