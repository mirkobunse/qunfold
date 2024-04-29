import numpy as np
from copy import deepcopy
from .methods import AbstractMethod
from .transformers import AbstractTransformer, class_prevalences, _check_y

class AverageVoting(AbstractMethod):
  """An ensemble of quantification methods the predictions of which are averaged.

  Args:
      base_method: The quantification method (from which to create n_estimators ensemble members) or a list of these members.
      n_estimators (optional): The number of ensemble members. Must be specified if base_method is not a list. Defaults to None.
      training_strategy (optional): How to train each ensemble member. Must be either "full" to use all data or "app" to train with random class prevalences. Defaults to "full".
  """
  def __init__(self, base_method, n_estimators=None, training_strategy="full"):
    self.base_method = base_method
    self.n_estimators = n_estimators
    self.training_strategy = training_strategy
  def fit(self, X, y, n_classes=None):
    pass # TODO
    return self
  def predict(self, X):
    return None # TODO

class EnsembleTransformer(AbstractTransformer):
  """An ensemble of transformers of which the outputs are concatenated.

  Args:
      base_transformer: Either a feature transformation (from which to create n_estimators ensemble members) or a list of these members.
      n_estimators (optional): The number of ensemble members. Must be specified if base_transformer is not a list. Defaults to None.
      training_strategy (optional): How to train each ensemble member. Must be either "full" to use all data or "app" to train with random class prevalences. Defaults to "full".
  """
  def __init__(self, base_transformer, n_estimators=None, training_strategy="full", random_state=None):
    self.base_transformer = base_transformer
    self.n_estimators = n_estimators
    self.training_strategy = training_strategy
    self.random_state = random_state
  def fit_transform(self, X, y, average=True, n_classes=None):
    if not average:
      raise ValueError("EnsembleTransformer does not support average=False")
    _check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    if isinstance(self.base_transformer, list):
      self.transformers_ = self.base_transformer
    elif self.n_estimators is not None:
      self.transformers_ = [
        deepcopy(self.base_transformer)
        for _ in range(self.n_estimators)
      ]
    else:
      raise ValueError("n_estimators must not be None if base_transformer is not a list")
    random_state = np.random.default_rng(self.random_state)
    Ms = [] # = (M_1, M_2, ...), the matrices of all ensemble members
    for transformer in self.transformers_:
      if self.training_strategy == "full":
        X_i, y_i = X, y # use all data for each transformer
      elif self.training_strategy == "app":
        X_i, y_i = subsample_app(X, y, n_classes, random_state)
      Ms.append(transformer.fit_transform(X_i, y_i, average, n_classes))
    return np.concatenate(Ms) # M = 1/n * (M_1, M_2, ...)
  def transform(self, X, average=True):
    if not average:
      raise ValueError("EnsembleTransformer does not support average=False")
    qs = [] # = (q_1, q_2, ...), the representations of all ensemble members
    for transformer in self.transformers_:
      qs.append(transformer.transform(X, average))
    return np.concatenate(qs) # q = 1/n * (q_1, q_2, ...)

def subsample_app(X, y, n_classes=None, random_state=None):
  """Subsample (X, y) according with random class proportions."""
  n_classes = len(class_prevalences(y, n_classes)) # not None anymore
  random_state = np.random.default_rng(random_state)
  i = draw_indices(
    y,
    random_state.dirichlet(np.ones(n_classes)),
    m = len(y),
    n_classes = n_classes,
    allow_duplicates = True,
    allow_others = False,
    allow_incomplete = False,
    min_samples_per_class = 1,
    random_state = random_state
  )
  return X[i], y[i]

######## (improved) code of git@github.com:mirkobunse/acspy.git ########

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
    to_take = np.maximum(np.round(m*p).astype(int), min_samples_per_class)
    if not allow_duplicates:
      to_take = np.minimum(m_pool, to_take)
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
