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
  """A simple model."""
  @nn.compact
  def __call__(self, x):
    # x = nn.Dense(features=300)(x)
    # x = nn.relu(x)
    # x = nn.Dense(features=300)(x)
    # x = nn.relu(x)
    x = nn.Dense(features=64)(x) # the LeQua T1B data has 28 classes
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

def mean_embedding(phi_X):
  # return nn.activation.softmax(phi_X, axis=1).mean(axis=0)
  # return phi_X.mean(axis=0)
  return nn.activation.sigmoid(phi_X).mean(axis=0)

class DeepTransformer(AbstractTransformer):
  def __init__(self, state):
    self.state = state
  def fit_transform(self, X, y, average=True):
    if not average:
      raise ValueError("DeepTransformer does not support average=False")
    self.p_trn = n_samples_per_class(y) / len(y) # also sets self.n_classes
    M = jnp.vstack([
      mean_embedding(self.state.apply_fn({ "params": self.state.params }, X_i))
      for X_i in [ X[y == i] for i in range(self.n_classes) ]
    ]).T
    return M
  def transform(self, X, average=True):
    if not average:
      raise ValueError("DeepTransformer does not support average=False")
    q = mean_embedding(self.state.apply_fn({ "params": self.state.params }, X))
    return q

@jax.jit
def training_step(state, sample):
  """Perform a single parameter update.

  Args:
      state: A TrainingState object.
      sample: A dict with keys "X_q", "X_M", and "p".

  Returns:
      An updated TrainingState object.
  """
  def loss_fn(params):
    qs = jnp.array([
      mean_embedding(state.apply_fn({ "params": params }, X_q))
      for X_q in sample["X_q"]
    ])
    M = jnp.vstack([
      mean_embedding(state.apply_fn({ "params": params }, X_i))
      for X_i in sample["X_M"]
    ]).T
    v = sample["p_Ts"] - (jnp.linalg.pinv(M) @ qs.T).T
    return (v**2).sum(axis=1).mean() # least squares ||p* - pinv(M)q||
  state = state.apply_gradients(grads=jax.grad(loss_fn)(state.params)) # update the state
  metric_updates = state.metrics.single_from_model_output(
    loss = loss_fn(state.params)
  )
  metrics = state.metrics.merge(metric_updates)
  return state.replace(metrics=metrics) # update the state with metrics

def main(
    n_batches = 500,
    batch_size = 64,
    sample_size = 1000,
    n_batches_between_evaluations = 10, # how many samples to process between evaluations
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
  print(f"[baseline] MAE={error:.5f}+-{error_std:.5f}")

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
    learning_rate = 1,
    momentum = .9,
  )

  # take out the training
  print("Training...")
  X_trn = jnp.array(X_trn, dtype=jnp.float32)
  y_trn = jnp.array(y_trn)
  sample_rng = np.random.default_rng(25)
  metrics_history = {
    "trn_loss": [],
  }
  for batch_index in range(n_batches):

    # evaluate every n_batches_between_evaluations
    if batch_index % n_batches_between_evaluations == 0 or (batch_index+1) == n_batches:
      quapy_method = QuaPyWrapper(qunfold.GenericMethod(
        qunfold.LeastSquaresLoss(),
        DeepTransformer(training_state)
      )).fit(trn_data)
      errors = qp.evaluation.evaluate( # errors of all predictions
        quapy_method,
        protocol = val_gen,
        error_metric = "ae"
      )
      error = errors.mean()
      error_std = errors.std()
      print(
        f"[{batch_index:2d}/{n_batches}] ",
        f"MAE={error:.5f}+-{error_std:.5f}",
        f", trn_loss={np.array(metrics_history['trn_loss'])[-n_batches_between_evaluations:].mean():e}" if batch_index > 0 else "",
        sep = "",
      )

    # update parameters and metrics
    p_Ts = sample_rng.dirichlet(np.ones(28), size=batch_size)
    sample = {
      "X_q": [
        X_trn[draw_indices(y_trn, p_T, sample_size, random_state=sample_rng)]
        for p_T in p_Ts
      ],
      "X_M": [ X_trn[y_trn == i] for i in range(28) ],
      "p_Ts": p_Ts,
    }
    training_state = training_step(training_state, sample)

    # compute average training metrics for this epoch and reset the metric state
    for metric, value in training_state.metrics.compute().items():
      metrics_history[f"trn_{metric}"].append(value)
    training_state = training_state.replace(metrics=training_state.metrics.empty()) # reset

if __name__ == "__main__":
  main()