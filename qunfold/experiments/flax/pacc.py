import argparse
import itertools
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import quapy as qp
from flax import linen as nn
from functools import partial
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from time import time
from . import (
  n_samples_per_class,
  draw_indices,
  SimpleModule,
  create_training_state,
  _solve_parallel,
)

# TODO implement a learning rate schedule with validation plateau-ing:
# https://github.com/HDembinski/essays/blob/master/regression.ipynb

def _main( # one trial of the experiment; to be taken out with multiple configurations
    args,
    n_experiments = None,
    n_epochs = 50,
    n_epochs_between_evaluations = 1, # how many samples to process between evaluations
    n_jobs = 8,
    n_val = 64,
  ):
  i_experiment, (n_features, learning_rate, batch_size) = args
  trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")
  X_trn, y_trn = trn_data.Xy
  X_trn = jnp.array(X_trn, dtype=jnp.float32)
  y_trn = jnp.array(y_trn)
  p_trn = n_samples_per_class(y_trn) / len(y_trn)
  avg_trn = np.zeros((28, len(y_trn))) # shape (n_classes, n_samples)
  for i in range(28):
    avg_trn[i, y_trn == i] = 1 / np.sum(y_trn == i)
  avg_trn = jnp.array(avg_trn, dtype=jnp.float32)
  val_gen.true_prevs.df = val_gen.true_prevs.df[:n_val] # use only some validation samples
  X_val = []
  p_val = []
  for X_i, p_i in val_gen():
    X_val.append(X_i)
    p_val.append(p_i)
  X_val = jnp.vstack(X_val)
  p_val = jnp.vstack(p_val)
  avg_val = np.zeros((p_val.shape[0], X_val.shape[0]))
  for i in range(p_val.shape[0]):
    avg_val[i,i*1000:(i+1)*1000] = 1 / 1000 # lequa uses 1000 items per sample
  avg_val = jnp.array(avg_val, dtype=jnp.float32)

  # instantiate the model
  module = SimpleModule(n_features)
  training_state = create_training_state(
    module,
    jax.random.key(0),
    learning_rate = learning_rate,
    momentum = .9,
  )

  # take out the training
  sample_rng = np.random.default_rng(25)
  metrics_history = {
    "trn_loss": [],
  }

  @jax.jit
  def training_step(state, X_batch, y_batch):
    """Perform a single parameter update.

    Returns:
        An updated TrainingState object.
    """
    def loss_fn(params):
      return optax.softmax_cross_entropy_with_integer_labels(
        logits = state.apply_fn({'params': params}, X_batch),
        labels = y_batch
      ).mean()
    loss, grad = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grad) # update the state
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    return state.replace(metrics=metrics) # update the state with metrics

  @jax.jit
  def validation_embedding(state):
    qs = jnp.dot(
      avg_val,
      nn.activation.softmax(state.apply_fn({ "params": state.params }, X_val), axis=1)
    )
    M = jnp.dot(
      avg_trn,
      nn.activation.softmax(state.apply_fn({ "params": state.params }, X_trn), axis=1)
    ).T
    return qs, M

  results = []
  t_0 = time()
  for epoch_index in range(n_epochs):
    if epoch_index == 1:
      t_0 = time() # reset after the first batch to ignore JIT overhead

    # update parameters and metrics
    i_epoch = np.random.default_rng(epoch_index).permutation(len(y_trn))
    for batch_index in range(len(y_trn) // batch_size):
      i_batch = i_epoch[batch_index * batch_size:(batch_index+1) * batch_size]
      training_state = training_step(training_state, X_trn[i_batch], y_trn[i_batch])

    # compute average training metrics for this epoch and reset the metric state
    for metric, value in training_state.metrics.compute().items():
      metrics_history[f"trn_{metric}"].append(value)
    training_state = training_state.replace(metrics=training_state.metrics.empty()) # reset

    # evaluate every n_epochs_between_evaluations
    if (epoch_index+1) % n_epochs_between_evaluations == 0 or epoch_index == 0:
      qs, M = validation_embedding(training_state)
      solve_parallel = partial(_solve_parallel, M=M)
      errors = []
      with Pool(n_jobs if n_jobs > 0 else None) as pool:
        errors.extend(pool.imap(solve_parallel, zip(qs, p_val)))
      error = np.mean(errors)
      error_std = np.std(errors)
      trn_loss = np.array(metrics_history["trn_loss"])[-n_epochs_between_evaluations:].mean()
      it_per_s = max(epoch_index, 1) / (time() - t_0)
      results.append({
        "n_features": str(n_features),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epoch_index": epoch_index+1,
        "mae": error,
        "mae_std": error_std,
        "trn_loss": trn_loss,
        "it_per_s": it_per_s,
      })
      print(
        f"[{i_experiment+1}/{n_experiments} |",
        f"{epoch_index+1}/{n_epochs}]",
        f"MAE={error:.5f}+-{error_std:.5f},",
        f"trn_loss={trn_loss:e},",
        f"{it_per_s:.3f} it/s",
      )
  return results

def main(
    output_path,
    is_test_run = False
  ):
  # configure the experiments
  n_features = [ [28], [64, 28], [128, 28], [64, 64, 28], [128, 128, 28] ]
  learning_rates = [ 1e-4, 1e-3, 1e-2, 1e-1 ]
  batch_sizes = [ 32, 64, 128 ]
  kwargs = {}
  if is_test_run:
    n_features = [ [28], [64, 28] ]
    learning_rates = [ 1e-2 ]
    batch_sizes = [ 8 ]
    kwargs["n_epochs"] = 2
    kwargs["n_epochs_between_evaluations"] = 2
    kwargs["n_val"] = 3
  kwargs["n_experiments"] = len(n_features) * len(learning_rates) * len(batch_sizes)

  # run all experiments one after another
  print(f"Starting {kwargs['n_experiments']} experiments")
  results = []
  for args in enumerate(itertools.product(n_features, learning_rates, batch_sizes)):
    results.extend(_main(args, **kwargs))
  results = pd.DataFrame(results)

  # store the results
  results.to_csv(output_path)
  print(f"{results.shape[0]} results succesfully stored at {output_path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('output_path', type=str, help='path of an output *.csv file')
  parser.add_argument("--is_test_run", action="store_true")
  args = parser.parse_args()
  main(
      args.output_path,
      args.is_test_run,
  )
