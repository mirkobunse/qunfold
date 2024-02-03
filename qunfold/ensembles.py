import numpy as np
from copy import deepcopy
from .methods import AbstractMethod
from .transformers import AbstractTransformer, class_prevalences, _check_y

class AverageVoting(AbstractMethod):
  """An ensemble of quantification methods the predictions of which are averaged.

  Args:
      base_method: The quantification method from which to create ensemble members.
      n_estimators: The number of ensemble members.
  """
  def __init__(self, base_method, n_estimators):
    self.base_method = base_method
    self.n_estimators = n_estimators
  def fit(self, X, y, n_classes=None):
    pass # TODO
    return self
  def predict(self, X):
    return None # TODO

class EnsembleTransformer(AbstractTransformer):
  """An ensemble of transformers of which the outputs are concatenated.

  Args:
      base_transformer: Either a feature transformation from which to create n_estimators ensemble members or a list of these members.
      n_estimators (optional): The number of ensemble members. Must be specified if base_transformer is not a list. Defaults to None.
  """
  def __init__(self, base_transformer, n_estimators=None):
    self.base_transformer = base_transformer
    self.n_estimators = n_estimators
  def fit_transform(self, X, y, average=True, n_classes=None):
    if not average:
      raise ValueError("EnsembleTransformer does not support average=False")
    _check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    n_classes = len(self.p_trn) # not None anymore
    if isinstance(self.base_transformer, list):
      self.transformers_ = self.base_transformer
    elif self.n_estimators is None:
      raise ValueError("n_estimators must not be None if base_transformer is not a list")
    else:
      self.transformers_ = [
        deepcopy(self.base_transformer)
        for _ in range(self.n_estimators)
      ]
    Ms = [] # = (M_1, M_2, ...), the matrices of all ensemble members
    for transformer in self.transformers_:
      # TODO subsample (X, y) through Bagging + APP
      transformer.fit_transform(X, y, average, n_classes)
      Ms.append(transformer.fit_transform(X, y, average, n_classes))
    return np.concatenate(Ms) # M = (M_1, M_2, ...)
  def transform(self, X, average=True):
    if not average:
      raise ValueError("EnsembleTransformer does not support average=False")
    qs = [] # = (q_1, q_2, ...), the representations of all ensemble members
    for transformer in self.transformers_:
      qs.append(transformer.transform(X, average))
    return np.concatenate(qs) # q = (q_1, q_2, ...)
