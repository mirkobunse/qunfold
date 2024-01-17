import argparse
import itertools
import jax
import jax.numpy as jnp
import numpy as np
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
  _draw_indices_parallel,
  _solve_parallel,
)

# TODO implement a learning rate schedule with validation plateau-ing:
# https://github.com/HDembinski/essays/blob/master/regression.ipynb
#
# Actually, before using an adaptive schedule, start with piecewise constant learning rates:
# https://coderzcolumn.com/tutorials/artificial-intelligence/optax-learning-rate-schedules-for-flax-jax-networks

# TODO use a magnitude-independent loss function like cosine distance
# instead of least squares -> make up for errors of the pinv solution

def _main( # one trial of the experiment; to be taken out with multiple configurations
    args,
    n_experiments = None,
    n_batches = 300,
    sample_size = 1000,
    n_batches_between_evaluations = 10, # how many samples to process between evaluations
    n_jobs = 8,
    n_val = 64,
  ):
  i_experiment, (n_features, lr_init, lr_shrinkage, batch_size) = args
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
    lr_init = lr_init,
    lr_steps = [100, 200], # assuming n_batches = 300; TODO generalize
    lr_shrinkage = lr_shrinkage,
    momentum = .9,
    rng = jax.random.key(0),
  )

  # instantiate training utilities
  avg_q = np.zeros((batch_size, batch_size * sample_size))
  for i in range(batch_size):
    avg_q[i,i*sample_size:(i+1)*sample_size] = 1 / sample_size
  avg_q = jnp.array(avg_q, dtype=jnp.float32)
  sample_rng = np.random.default_rng(25)
  metrics_history = {
    "trn_loss": [],
  }

  @jax.jit
  def training_step(state, p_Ts, X_q_i):
    """Perform a single parameter update.

    Returns:
        An updated TrainingState object.
    """
    def loss_fn(params):
      qs = jnp.dot(avg_q, nn.activation.sigmoid(state.apply_fn({ "params": params }, X_q_i)))
      M = jnp.dot(avg_M, nn.activation.sigmoid(state.apply_fn({ "params": params }, X_M))).T
      v = p_Ts - (jnp.linalg.pinv(M) @ qs.T).T # p_T - p_hat for each p_T
      return (v**2).sum(axis=1).mean() # least squares ||p* - pinv(M)q||
    loss, grad = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grad) # update the state
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    return state.replace(metrics=metrics) # update the state with metrics

  @jax.jit
  def validation_embedding(state):
    qs = jnp.dot(
      avg_val,
      nn.activation.sigmoid(state.apply_fn({ "params": state.params }, X_val))
    )
    M = jnp.dot(
      avg_trn,
      nn.activation.sigmoid(state.apply_fn({ "params": state.params }, X_trn))
    ).T
    return qs, M

  # take out the training
  results = []
  t_0 = time()
  for batch_index in range(n_batches):
    if batch_index == 1:
      t_0 = time() # reset after the first batch to ignore JIT overhead

    # draw a new (M, q) split for each step
    X_M, X_q, y_M, y_q = train_test_split(
      X_trn,
      y_trn,
      stratify = y_trn,
      test_size = .5,
      random_state = batch_index
    )
    avg_M = np.zeros((28, len(y_M))) # shape (n_classes, n_samples)
    for i in range(28):
      avg_M[i, y_M == i] = 1 / np.sum(y_M == i)
    avg_M = jnp.array(avg_M, dtype=jnp.float32)

    # update parameters and metrics
    p_Ts = sample_rng.dirichlet(np.ones(28), size=batch_size)
    q_i = []
    draw_indices_parallel = partial(_draw_indices_parallel, y=y_q, m=sample_size)
    with Pool(n_jobs if n_jobs > 0 else None) as pool:
      q_i.extend(pool.imap(draw_indices_parallel, enumerate(p_Ts)))
    p_Ts = jnp.vstack([ n_samples_per_class(y_q[i], 28) / sample_size for i in q_i ])
    X_q_i = jnp.vstack([ X_q[i] for i in q_i ])
    training_state = training_step(training_state, p_Ts, X_q_i)

    # compute average training metrics for this epoch and reset the metric state
    for metric, value in training_state.metrics.compute().items():
      metrics_history[f"trn_{metric}"].append(value)
    training_state = training_state.replace(metrics=training_state.metrics.empty()) # reset

    # evaluate every n_batches_between_evaluations
    if (batch_index+1) % n_batches_between_evaluations == 0 or batch_index == 0:
      qs, M = validation_embedding(training_state)
      solve_parallel = partial(_solve_parallel, M=M)
      errors = []
      with Pool(n_jobs if n_jobs > 0 else None) as pool:
        errors.extend(pool.imap(solve_parallel, zip(qs, p_val)))
      error = np.mean(errors)
      error_std = np.std(errors)
      trn_loss = np.array(metrics_history["trn_loss"])[-n_batches_between_evaluations:].mean()
      it_per_s = max(batch_index, 1) / (time() - t_0)
      results.append({
        "n_features": str(n_features),
        "lr_init": lr_init,
        "lr_shrinkage": lr_shrinkage,
        "batch_size": batch_size,
        "batch_index": batch_index+1,
        "mae": error,
        "mae_std": error_std,
        "trn_loss": trn_loss,
        "it_per_s": it_per_s,
      })
      print(
        f"[{i_experiment+1}/{n_experiments} |",
        f"{batch_index+1}/{n_batches}]",
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
  n_features = [ [64], [512], [1024] ]
  lr_init = [ 1e0, 1e1 ]
  lr_shrinkage = [ 1., .1 ] # no shrinkage vs considerable shrinkage (.1 and .5 seem similar)
  batch_size = [ 64 ] # does not seem to make a big difference (also tried 32, 128)
  kwargs = {}
  if is_test_run:
    n_features = [ [64], [128] ]
    lr_init = [ 1e1 ]
    lr_shrinkage = [ .1 ]
    batch_size = [ 8 ]
    kwargs["n_batches"] = 2
    kwargs["n_batches_between_evaluations"] = 2
    kwargs["n_val"] = 3
  kwargs["n_experiments"] = len(n_features)*len(lr_init)*len(lr_shrinkage)*len(batch_size)

  # run all experiments one after another
  print(f"Starting {kwargs['n_experiments']} experiments")
  results = []
  for args in enumerate(itertools.product(n_features, lr_init, lr_shrinkage, batch_size)):
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
