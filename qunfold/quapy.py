import inspect
from collections import defaultdict
from quapy.method.base import BaseQuantifier
from . import LinearMethod

#
# _get_params and _set_params use inspection to provide a functionality
# that is equivalent to the functionality of sklearns BaseEstimator,
# without the need to subtype it.
#
# https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/base.py#L112
#

def _get_param_names(cls):
    return sorted([ # collect the constructor parameters
        p.name
        for p in inspect.signature(cls.__init__).parameters.values()
        if p.name != "self" and p.kind not in [p.VAR_KEYWORD, p.VAR_POSITIONAL]
    ])

def _get_params(x, deep=True, _class=None):
    params = {}
    if _class is None:
        _class = x.__class__
    for k in _get_param_names(_class):
        v = getattr(x, k)
        if deep and not isinstance(v, type):
            if hasattr(v, "get_params"):
                deep_params = v.get_params()
            else:
                deep_params = _get_params(v, deep=True)
            params.update((k + "__" + k2, v2) for k2, v2 in deep_params.items())
        params[k] = v
    return params

def _set_params(x, valid_params, **params):
    if not params:
        return x
    if valid_params is None:
        if hasattr(x, "get_params"):
            valid_params = x.get_params(deep=True)
        else:
            valid_params = _get_params(x, deep=True)
    nested_params = defaultdict(dict)
    for key, value in params.items():
        key, delim, sub_key = key.partition("__")
        if key not in valid_params:
            raise ValueError(f"Invalid parameter {key!r} for {x}.")
        if delim:
            nested_params[key][sub_key] = value
        else:
            setattr(x, key, value)
            valid_params[key] = value
    for key, sub_params in nested_params.items():
        if hasattr(valid_params[key], "set_params"):
            valid_params[key].set_params(**sub_params)
        else:
            _set_params(valid_params[key], None, **sub_params)
    return x

class QuaPyWrapper(BaseQuantifier):
    """A thin wrapper for using qunfold methods in QuaPy.

    Args:
        generic_method: A LinearMethod method to wrap.

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
    def __init__(self, generic_method):
        self.generic_method = generic_method
    def fit(self, data): # data : LabelledCollection
        self.generic_method.fit(*data.Xy, data.n_classes)
        return self
    def quantify(self, X):
        return self.generic_method.predict(X)
    def set_params(self, **params):
        _set_params(self.generic_method, self.get_params(deep=True), **params)
        return self
    def get_params(self, deep=True):
        return _get_params(self.generic_method, deep, LinearMethod)
