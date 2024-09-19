from abc import ABC, abstractmethod

class AbstractMethod(ABC):
  """Abstract base class for quantification methods."""
  @abstractmethod
  def fit(self, X, y, n_classes=None):
    """Fit this quantifier to data.

    Args:
        X: The feature matrix to which this quantifier will be fitted.
        y: The labels to which this quantifier will be fitted.
        n_classes (optional): The number of expected classes. Defaults to `None`.

    Returns:
        This fitted quantifier itself.
    """
    pass
  @abstractmethod
  def predict(self, X):
    """Predict the class prevalences in a data set.

    Args:
        X: The feature matrix for which this quantifier will make a prediction.

    Returns:
        A numpy array of class prevalences.
    """
    pass
