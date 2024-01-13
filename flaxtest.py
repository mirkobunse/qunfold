import flax # neural networks with JAX
import jax
import jax.numpy as jnp
import numpy as np
import optax # common loss functions and optimizers
import quapy as qp
import qunfold
from copy import deepcopy
from clu import metrics # utilities for handling metrics in training loops
from flax import linen as nn
from flax.training import train_state
from functools import partial
from multiprocessing import Pool
from qunfold.quapy import QuaPyWrapper
from qunfold.transformers import AbstractTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from time import time
from typing import Callable, Sequence



######## code of git@github.com:mirkobunse/acspy.git ########

def n_samples_per_class(y, n_classes=None):
  """Determine the number of instances per class.

  Args:
      y: An array of labels, shape (n_samples,).
      n_classes (optional): The number of classes. Defaults to `None`, which corresponds to `len(np.unique(y))`.

  Returns:
      An array of label counts, shape (n_classes,).
  """
  if n_classes is None:
    n_classes = len(np.unique(y))
  n_samples_per_class = np.zeros(n_classes, dtype=int)
  i, n = np.unique(y, return_counts=True)
  n_samples_per_class[i] = n # non-existing classes maintain a zero entry
  return n_samples_per_class

def draw_indices(
    y,
    score,
    m = None,
    n_classes = None,
    allow_duplicates = False,
    allow_others = True,
    allow_incomplete = True,
    min_samples_per_class = 0,
    random_state = None,
  ):
  """Draw m indices of instances, according to the given class scores.

  Args:
      y: An pool of labels, shape (n_samples,).
      score: An array of utility scores for the acquisition of each clas, shape (n_classes,).
      m (optional): The total number of instances to be drawn. Defaults to `None`, which is only valid for binary classification tasks and which corresponds to drawing the maximum number of instances for which the scores can be fulfilled.
      n_classes (optional): The number of classes. Defaults to `None`, which corresponds to `len(np.unique(y))`.
      allow_duplicates (optional): Whether to allow drawing one sample multiple times. Defaults to `False`.
      allow_others (optional): Whether to allow drawing other classes if the desired class is exhausted. Defaults to `True`.
      allow_incomplete (optional): Whether to allow an incomplete draw if all classes are exhausted. Defaults to `True`.
      min_samples_per_class (optional): The minimum number of samples per class. Defaults to `0`.
      random_state (optional): A numpy random number generator, or a seed thereof. Defaults to `None`, which corresponds to `np.random.default_rng()`.

  Returns:
      An array of indices, shape (m,).
  """
  if n_classes is None:
    n_classes = len(np.unique(y))
  elif len(score) != n_classes:
    raise ValueError("len(score) != n_classes")
  random_state = np.random.default_rng(random_state)
  m_pool = n_samples_per_class(y, n_classes) # number of available samples per class
  p = np.maximum(0, score) / np.maximum(0, score).sum() # normalize scores to probabilities
  if not np.isfinite(p).all():
    raise ValueError(f"NaN probabilities caused by score={score}")

  # determine the number of instances to acquire from each class
  if m is not None:
    to_take = np.minimum(m_pool, np.maximum(np.round(m*p).astype(int), min_samples_per_class))
    while m != to_take.sum(): # rarely takes more than one iteration
      m_remaining = m - to_take.sum()
      if m_remaining > 0: # are additional draws needed?
        i = np.nonzero(to_take < m_pool)[0]
        if len(i) == 0:
          if allow_incomplete:
            break
          else:
            raise ValueError(
              f"All classes are exhausted; consider setting allow_incomplete=True"
            )
        elif allow_others or len(i) == len(p):
          bincount = np.bincount(random_state.choice(
            len(i),
            size = m_remaining,
            p = np.maximum(1/m, p[i]) / np.maximum(1/m, p[i]).sum(),
          ))
          to_take[i[:len(bincount)]] += bincount
        else:
          raise ValueError(
            f"Class {np.setdiff1d(np.arange(len(p)), i)[0]} exhausted; "
            "consider setting allow_others=True "
          )
      elif m_remaining < 0: # are less draws needed?
        i = np.nonzero(to_take > min_samples_per_class)[0]
        bincount = np.bincount(random_state.choice(
          len(i),
          size = -m_remaining, # negate to get a positive value
          p = np.maximum(1/m, 1-p[i]) / np.maximum(1/m, 1-p[i]).sum(),
        ))
        to_take[i[:len(bincount)]] -= bincount
      to_take = np.maximum(min_samples_per_class, np.minimum(to_take, m_pool))
  else: # m is None
    if len(m_pool) != 2:
      raise ValueError("m=None is only valid for binary classification tasks")
    to_take = m_pool.copy()
    if p[0] > m_pool[0] / m_pool.sum(): # do we have to increase the class "0" probability?
      to_take[1] = int(np.round(m_pool[0] * p[1] / p[0])) # sub-sample class "1"
    else: # otherwise, we need to increase the class "1" probability
      to_take[0] = int(np.round(m_pool[1] * p[0] / p[1])) # sub-sample class "0"
    to_take = np.minimum(to_take, m_pool)

  # draw indices that represent instances of the classes to acquire
  i_rand = random_state.permutation(len(y)) # random order after shuffling
  i_draw = [] # array of drawn index arrays (relative to i_rand)
  for c in range(n_classes):
    i_c = np.arange(len(y))[y[i_rand] == c]
    if to_take[c] <= len(i_c):
      i_draw.append(i_c[:to_take[c]])
    elif allow_duplicates:
      i_c = np.tile(i_c, int(np.ceil(to_take[c] / len(i_c)))) # repeat i_c multiple times
      i_draw.append(i_c[:to_take[c]]) # draw from the tiled array
    else:
      raise ValueError(
        f"Class {c} exhausted; consider setting allow_duplicates=True "
        f"({to_take[c]} requested, {len(i_c)} available)"
      )
  return i_rand[np.concatenate(i_draw)]

######## end of git@github.com:mirkobunse/acspy.git  ########



def draw_indices_parallel(args, y, m):
  return draw_indices(y, args[1], m=m, random_state=args[0])

def predict_parallel(args, M): # args = (q, p)
  return qp.error.ae(
    args[1],
    qunfold.GenericMethod(qunfold.LeastSquaresLoss(), None).solve(args[0], M)
  )


class SimpleModule(nn.Module):
  """A simple dense neural network with ReLU activation."""
  n_features: Sequence[int]
  @nn.compact
  def __call__(self, x):
    for i, n_features in enumerate(self.n_features):
      x = nn.Dense(n_features)(nn.activation.relu(x) if i > 0 else x)
    return x

# a storage for all state involved in training, including metrics
@flax.struct.dataclass
class Metrics(metrics.Collection):
  loss: metrics.Average.from_output("loss") # average loss
class TrainingState(train_state.TrainState):
  metrics: Metrics

def create_training_state(module, rng, learning_rate, momentum):
  """Create an initial `TrainingState`."""
  return TrainingState.create(
    apply_fn = module.apply,
    params = module.init( # initialize parameters
      rng,
      jnp.ones((1, 300)) # a template batch with one sample, 300 dimensions
    )["params"],
    tx = optax.sgd(learning_rate, momentum),
    metrics = Metrics.empty()
  )

def mean_embedding(phi_X, use_classifier):
  if use_classifier:
    return nn.activation.softmax(phi_X, axis=1).mean(axis=0)
  return nn.activation.sigmoid(phi_X).mean(axis=0)

# TODO implement a learning rate schedule with validation plateau-ing:
# https://github.com/HDembinski/essays/blob/master/regression.ipynb

@jax.jit
def training_step_classifier(state, X, y):
  def loss_fn(params):
    return optax.softmax_cross_entropy_with_integer_labels(
      logits = state.apply_fn({'params': params}, X),
      labels = y
    ).mean()
  state = state.apply_gradients(grads=jax.grad(loss_fn)(state.params))
  metric_updates = state.metrics.single_from_model_output(
    loss = loss_fn(state.params)
  )
  metrics = state.metrics.merge(metric_updates)
  return state.replace(metrics=metrics) # update the state with metrics

def training_epoch_classifier(state, epoch_index, X, y, batch_size=128):
  """Take out an epoch for the classification head."""
  i_epoch = np.random.default_rng(epoch_index).permutation(len(y))
  for batch_index in range(len(y) // batch_size):
    i_batch = i_epoch[batch_index * batch_size:(batch_index+1) * batch_size]
    state = training_step_classifier(state, X[i_batch], y[i_batch])
  return state

def main(
    n_batches = 500,
    batch_size = 64,
    sample_size = 1000,
    n_batches_between_evaluations = 10, # how many samples to process between evaluations
    use_classifier = False,
    n_jobs = 8,
  ):
  trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")
  X_trn, y_trn = trn_data.Xy
  p_trn = n_samples_per_class(y_trn) / len(y_trn)
  val_gen.true_prevs.df = val_gen.true_prevs.df[:n_jobs] # use only n_jobs validation samples

  # baseline performance: SLD, the winner @ LeQua2022
  baseline = qp.method.aggregative.EMQ(LogisticRegression(C=0.01)).fit(trn_data)
  errors = qp.evaluation.evaluate( # errors of all predictions
    baseline,
    protocol = val_gen,
    error_metric = "ae"
  )
  error = errors.mean()
  error_std = errors.std()
  print(f"[baseline (SLD)] MAE={error:.5f}+-{error_std:.5f}")

  # instantiate the model
  module = SimpleModule([28] if use_classifier else [64])
  print(module.tabulate( # inspect the structure of the model
    jax.random.key(0),
    jnp.ones((1, 300)),
    compute_flops = True,
    compute_vjp_flops = True
  ))
  training_state = create_training_state(
    module,
    jax.random.key(0),
    learning_rate = 1e-2 if use_classifier else 1,
    momentum = .9,
  )

  # take out the training
  print("Training...")
  X_trn = jnp.array(X_trn, dtype=jnp.float32)
  y_trn = jnp.array(y_trn)
  X_M, X_q, y_M, y_q = train_test_split(
    X_trn,
    y_trn,
    stratify = y_trn,
    test_size = .5,
    random_state = 25
  )
  avg_M = np.zeros((28, len(X_M))) # shape (n_classes, n_samples)
  for i in range(28):
    avg_M[i, y_M == i] = 1 / np.sum(y_M == i)
  avg_q = np.zeros((batch_size, batch_size * sample_size))
  for i in range(batch_size):
    avg_q[i,i*sample_size:(i+1)*sample_size] = 1 / sample_size
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

  t_0 = time()
  for batch_index in range(n_batches):
    if batch_index == 1:
      t_0 = time() # reset after the first batch to ignore JIT overhead

    # evaluate every n_batches_between_evaluations
    if batch_index % n_batches_between_evaluations == 0 or (batch_index+1) == n_batches:
      errors = []
      qps = [ # (q, p)
        (mean_embedding(training_state.apply_fn({ "params": training_state.params }, X_i), use_classifier), p_i)
        for (X_i, p_i) in val_gen()
      ]
      _predict_parallel = partial(
        predict_parallel,
        M = np.vstack([
          mean_embedding(training_state.apply_fn({ "params": training_state.params }, X_i), use_classifier)
          for X_i in [ X_trn[y_trn == i] for i in range(28) ]
        ]).T
      )
      with Pool(n_jobs if n_jobs > 0 else None) as pool:
        errors.extend(pool.imap(_predict_parallel, qps))
      error = np.mean(errors)
      error_std = np.std(errors)
      print(
        f"[{batch_index:2d}/{n_batches}] ",
        f"MAE={error:.5f}+-{error_std:.5f}",
        f", trn_loss={np.array(metrics_history['trn_loss'])[-n_batches_between_evaluations:].mean():e}" if batch_index > 0 else "",
        f", {(batch_index-1) / (time() - t_0):.3f} it/s" if batch_index > 0 else "",
        sep = "",
      )

    # update parameters and metrics
    if use_classifier:
      training_state = training_epoch_classifier(training_state, batch_index, X_trn, y_trn)
    else:
      p_Ts = sample_rng.dirichlet(np.ones(28), size=batch_size)
      q_i = []
      draw_parallel = partial(draw_indices_parallel, y=y_q, m=sample_size)
      with Pool(n_jobs if n_jobs > 0 else None) as pool:
        q_i.extend(pool.imap(draw_parallel, enumerate(p_Ts)))
      p_Ts = jnp.vstack([ n_samples_per_class(y_q[i], 28) / sample_size for i in q_i ])
      X_q_i = jnp.vstack([ X_q[i] for i in q_i ])
      training_state = training_step(training_state, p_Ts, X_q_i)

    # compute average training metrics for this epoch and reset the metric state
    for metric, value in training_state.metrics.compute().items():
      metrics_history[f"trn_{metric}"].append(value)
    training_state = training_state.replace(metrics=training_state.metrics.empty()) # reset

if __name__ == "__main__":
  main()
