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
from qunfold.quapy import QuaPyWrapper
from qunfold.transformers import AbstractTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from time import time



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



class SimpleModule(nn.Module):
  """A simple neural network."""
  @nn.compact
  def __call__(self, x):
    # x = nn.Dense(features=300)(x)
    # x = nn.relu(x)
    return (
      nn.Dense(features=64)(x), # the typical output
      nn.Dense(features=28)(x), # an additional output for 28 classes
    )

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
    return nn.activation.softmax(phi_X[1], axis=1).mean(axis=0)
  return nn.activation.sigmoid(phi_X[0]).mean(axis=0)

class DeepTransformer(AbstractTransformer):
  def __init__(self, state, use_classifier):
    self.state = state
    self.use_classifier = use_classifier
  def fit_transform(self, X, y, average=True, n_classes=None):
    if not average:
      raise ValueError("DeepTransformer does not support average=False")
    qunfold.transformers._check_y(y, n_classes)
    self.p_trn = qunfold.transformers.class_prevalences(y, n_classes)
    M = jnp.vstack([
      mean_embedding(
        self.state.apply_fn({ "params": self.state.params }, X_i),
        self.use_classifier
      )
      for X_i in [ X[y == i] for i in range(len(self.p_trn)) ]
    ]).T
    return M
  def transform(self, X, average=True):
    if not average:
      raise ValueError("DeepTransformer does not support average=False")
    q = mean_embedding(
      self.state.apply_fn({ "params": self.state.params }, X),
      self.use_classifier
    )
    return q

# TODO implement a learning rate schedule with validation plateau-ing:
# https://github.com/HDembinski/essays/blob/master/regression.ipynb

@jax.jit
def training_step_classifier(state, X, y):
  def loss_fn(params):
    return optax.softmax_cross_entropy_with_integer_labels(
      logits = state.apply_fn({'params': params}, X)[1],
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
  ):
  trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")
  X_trn, y_trn = trn_data.Xy
  p_trn = n_samples_per_class(y_trn) / len(y_trn)
  val_gen.true_prevs.df = val_gen.true_prevs.df[:3] # use only 3 validation samples

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
  module = SimpleModule()
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
      qs = jnp.dot(avg_q, nn.activation.sigmoid(state.apply_fn({ "params": params }, X_q_i)[0]))
      M = jnp.dot(avg_M, nn.activation.sigmoid(state.apply_fn({ "params": params }, X_M)[0])).T
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
      t_eval = time() # remember the evaluation time to ignore it later
      quapy_method = QuaPyWrapper(qunfold.GenericMethod(
        qunfold.LeastSquaresLoss(),
        DeepTransformer(training_state, use_classifier)
      )).fit(trn_data)
      errors = qp.evaluation.evaluate( # errors of all predictions
        quapy_method,
        protocol = val_gen,
        error_metric = "ae"
      )
      error = errors.mean()
      error_std = errors.std()
      t_0 += time() - t_eval # ignore evaluation time by pretending to have started later
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
      X_q_i = jnp.vstack([
        X_q[draw_indices(y_q, p_T, sample_size, random_state=sample_rng)]
        for p_T in p_Ts
      ])
      training_state = training_step(training_state, p_Ts, X_q_i)

    # # do we need a softmax operator for pinv(M) @ q ?
    # _qs = jnp.array([
    #   mean_embedding(training_state.apply_fn({ "params": training_state.params }, X_q))
    #   for X_q in sample["X_q"]
    # ])
    # _M = jnp.vstack([
    #   mean_embedding(training_state.apply_fn({ "params": training_state.params }, X_i))
    #   for X_i in sample["X_M"]
    # ]).T
    # _p_hat = (jnp.linalg.pinv(_M) @ _qs.T).T
    # _p_hat = _p_hat / _p_hat.sum(axis=1, keepdims=True)
    # print(_p_hat.sum(axis=1))

    # # are we allowed to back-propagate through pinv(M)? M is required to have a constant rank
    # _M = jnp.vstack([
    #   mean_embedding(training_state.apply_fn({ "params": training_state.params }, X_i))
    #   for X_i in sample["X_M"]
    # ]).T
    # print(jnp.linalg.matrix_rank(_M))

    # compute average training metrics for this epoch and reset the metric state
    for metric, value in training_state.metrics.compute().items():
      metrics_history[f"trn_{metric}"].append(value)
    training_state = training_state.replace(metrics=training_state.metrics.empty()) # reset

if __name__ == "__main__":
  main()
