import inspect
from collections import defaultdict

class BaseMixin():
  """
  A mix-in for any configurable component. This mix-in defines `get_params` and `set_params` for hyper-parameter optimization, a `clone` method, and a concise string representation through `__str__`.
  """

  @classmethod
  def _get_param_names(cls): # utility for get_params
    return sorted([ p.name for p in inspect.signature(cls).parameters.values() ])

  def get_params(self, deep=True):
    """
    Get the parameters of this component, just like in scikit-learn estimators.
    """
    params = dict()
    for k in self._get_param_names():
      v = getattr(self, k)
      if deep and hasattr(v, "get_params") and not isinstance(v, type):
        params.update((k + "__" + _k, _v) for _k, _v in v.get_params(deep).items())
      params[k] = v
    return params

  def set_params(self, **params):
    """
    Set the parameters of this component, just like in scikit-learn estimators.
    """
    valid_params = self.get_params(deep=True)
    nested_params = defaultdict(dict)
    for k, v in params.items():
      k, delim, k_nested = k.partition("__")
      if k not in valid_params:
        raise ValueError(f"Invalid parameter \"{k}\" for {self}")
      if delim:
        nested_params[k][k_nested] = v
      else:
        setattr(self, k, v)
        valid_params[k] = v
    for k, nested in nested_params.items():
        valid_params[k].set_params(**nested)
    return self

  def clone(self, **params):
    """
    Create a clone of this object. If additional keyword arguments are provided, set these values in the newly created clone.
    """
    clone_params = self.get_params(deep=False)
    for name, param in clone_params.items():
      if isinstance(param, BaseMixin): # deeply clone BaseMixin types
        clone_params[name] = param.clone()
    clone_params = clone_params | params # update with additional arguments
    return self.__class__(**clone_params)

  def __str__(self): # logging sugar: a concise string representation
    params = []
    for param in inspect.signature(self.__class__).parameters.values():
      if getattr(self, param.name) != param.default:
        params.append(f"{param.name}={getattr(self, param.name)}")
    return f"{self.__class__.__name__}({', '.join(params)})"

  # TODO add the @dataclass annotation to all sub-classes of BaseMixin
