import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import quapy as qp
import qunfold
from clu import metrics
from flax import linen as nn
from flax.training import train_state
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

def create_training_state(module, lr_init, lr_steps, lr_shrinkage, momentum, rng):
  """Create an initial `TrainingState`.

  Args:
      module: a Flax neural network module
      lr_init: the initial learning rate of learning rate-scheduled SGD
      lr_steps: a list of steps at which to reduce the learning rate
      lr_shrinkage: a factor with which the learning rate is multiplied at each step
      momentum: the SGD momentum
      rng: controls the random initialization of the parameters
  """
  return TrainingState.create(
    apply_fn = module.apply,
    params = module.init( # initialize parameters
      rng,
      jnp.ones((1, 300)) # a template batch with one sample, 300 dimensions
    )["params"],
    tx = optax.sgd(
      learning_rate = optax.piecewise_constant_schedule(
        init_value = lr_init,
        boundaries_and_scales = { x: lr_shrinkage for x in lr_steps }
      ),
      momentum = momentum
    ),
    metrics = Metrics.empty()
  )

def _draw_indices_parallel(args, y, m): # args = (seed, p_T)
  return draw_indices(y, args[1], m=m, random_state=args[0])

def _solve_parallel(args, M): # args = (q, p_true)
  p_est = qunfold.GenericMethod(qunfold.LeastSquaresLoss(), None).solve(args[0], M)
  return qp.error.ae(args[1], p_est)
