import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import unique_labels

class CVClassifier(BaseEstimator, ClassifierMixin):
  """An ensemble of classifiers that are trained from cross-validation folds.

  All objects of this type have a fixed attribute `oob_score = True` and, when trained, a fitted attribute `self.oob_decision_function_`, just like scikit-learn bagging classifiers.

  Args:
      estimator: A classifier that implements the API of scikit-learn.
      n_estimators (optional): The number of stratified cross-validation folds. Defaults to `5`.
      random_state (optional): The random state for stratification. Defaults to `None`.

  Examples:
      Here, we create an instance of ACC that trains a logistic regression classifier with 10 cross-validation folds.

          >>> ACC(CVClassifier(LogisticRegression(), 10))
  """
  def __init__(self, estimator, n_estimators=5, random_state=None):
    self.estimator = estimator
    self.n_estimators = n_estimators
    self.random_state = random_state
    self.oob_score = True # the whole point of this class is to have an oob_score
  def fit(self, X, y):
    self.estimators_ = []
    self.i_classes_ = [] # the indices of each estimator's subset of classes
    self.classes_ = unique_labels(y)
    self.oob_decision_function_ = np.zeros((len(y), len(self.classes_)))
    class_mapping = dict(zip(self.classes_, np.arange(len(self.classes_))))
    skf = StratifiedKFold(
        n_splits = self.n_estimators,
        random_state = self.random_state,
        shuffle = True
    )
    for i_trn, i_tst in skf.split(X, y):
      estimator = clone(self.estimator).fit(X[i_trn], y[i_trn])
      i_classes = np.array([ class_mapping[_class] for _class in estimator.classes_ ])
      y_pred = estimator.predict_proba(X[i_tst])
      self.oob_decision_function_[i_tst[:, np.newaxis], i_classes[np.newaxis, :]] = y_pred
      self.estimators_.append(estimator)
      self.i_classes_.append(i_classes)
    return self
  def predict_proba(self, X):
    if not hasattr(self, "classes_"):
      raise NotFittedError()
    y_pred = np.zeros((len(self.estimators_), len(X), len(self.classes_)))
    for i, (estimator, i_classes) in enumerate(zip(self.estimators_, self.i_classes_)):
      y_pred[i, :, i_classes] = estimator.predict_proba(X).T
    return np.mean(y_pred, axis=0) # shape (n_samples, n_classes)
  def predict(self, X):
    y_pred = self.predict_proba(X).argmax(axis=1) # class indices
    return self.classes_[y_pred]
