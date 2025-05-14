import inspect
from dataclasses import dataclass
from quapy.method.base import BaseQuantifier
from . import AbstractMethod
from .base import BaseMixin


@dataclass
class QuaPyWrapper(BaseQuantifier, BaseMixin):
    """A thin wrapper for using qunfold methods in QuaPy.

    Args:
      _method: An instance of `qunfold.methods.AbstractMethod` to wrap.

    Examples:
      Here, we wrap an instance of ACC to perform a grid search with QuaPy.

        >>> qunfold_method = QuaPyWrapper(ACC(RandomForestClassifier(obb_score=True)))
        >>> quapy.model_selection.GridSearchQ(
        >>>     model = qunfold_method,
        >>>     param_grid = { # try both splitting criteria
        >>>         "representation__classifier__estimator__criterion": ["gini", "entropy"],
        >>>     },
        >>>     # ...
        >>> )
    """
    _method: AbstractMethod

    def fit(self, X, y):
        self._method.fit(X, y, n_classes=len(set(y)))
        return self

    def predict(self, X):
        return self._method.predict(X)

    def set_params(self, **params):
        self._method.set_params(**params)
        return self

    def get_params(self, deep=True):
        return self._method.get_params(deep)

    def __str__(self):
        return self._method.__str__()

