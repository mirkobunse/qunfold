# API

<<<<<<< HEAD
The `AbstractMethod` defines the interface for all quantification and unfolding algorithms. Most importantly, this interface consists of their `fit` and `predict` methods.
=======
The `AbstractMethod` defines the interface of all quantification and unfolding algorithms. Most importantly, this interface consists of their `fit` and `predict` methods.
>>>>>>> upstream/main

```{eval-rst}
.. autoclass:: qunfold.AbstractMethod
   :members:
```

<<<<<<< HEAD
Instances of [](#popular-algorithms) for quantification and unfolding are created through the corresponding constructors. However, you can also define your own quantification methods as a `LinearMethod` that combines an arbitrary choice of [](#losses), [](#regularizers) and [](#feature-transformations).
=======
Instances of many [](#popular-algorithms) for quantification and unfolding are created through the corresponding constructors. However, you can also define your own quantification method as a `LinearMethod` that combines an arbitrary choice of [](#losses), [](#regularizers) and [](#data-representations).
>>>>>>> upstream/main

```{eval-rst}
.. autoclass:: qunfold.LinearMethod
   :members:
```


## Popular algorithms

We categorize existing, well-known quantification and unfolding algorithms into [](#classify-and-count) methods, [](#distribution-matching) methods, and [](#unfolding) methods. Each of these methods consists of a fixed combination of [](#losses), [](#regularizers), and [](#data-representations).


### Classify and count

```{eval-rst}
.. autoclass:: qunfold.ACC

.. autoclass:: qunfold.PACC
```


### Distribution matching

```{eval-rst}
.. autoclass:: qunfold.EDx

.. autoclass:: qunfold.EDy

.. autoclass:: qunfold.HDx

.. autoclass:: qunfold.HDy

.. autoclass:: qunfold.KMM
```


### Unfolding

```{eval-rst}
.. autoclass:: qunfold.RUN
```


### Methods beyond systems of linear equations

Not all quantification algorithms make predictions by solving systems of linear equations. Instead, the following methods maximize the likelihood of the prediction directly.

```{eval-rst}
.. autoclass:: qunfold.LikelihoodMaximizer

.. autoclass:: qunfold.ExpectationMaximizer
```


## Losses

Instances of `AbstractLoss` provide the loss functions for linear quantification methods. The `FunctionLoss`, also an abstract class, is a utility for creating such loss functions from JAX function objects.

```{eval-rst}
.. autoclass:: qunfold.AbstractLoss
   :members:

.. autoclass:: qunfold.FunctionLoss
   :members:
```

The following concrete sub-classes define the loss functions of existing methods.

```{eval-rst}
.. autoclass:: qunfold.LeastSquaresLoss

.. autoclass:: qunfold.EnergyLoss

.. autoclass:: qunfold.HellingerSurrogateLoss

.. autoclass:: qunfold.BlobelLoss

.. autoclass:: qunfold.CombinedLoss
```

```{hint}
You can use the `CombinedLoss` to create arbitrary, weighted sums of losses and regularizers.
```


## Regularizers

```{eval-rst}
.. autofunction:: qunfold.TikhonovRegularized

.. autoclass:: qunfold.TikhonovRegularization
```


## Data representations

Instances of `AbstractRepresentation` provide the data representations for linear quantification methods.

```{eval-rst}
.. autoclass:: qunfold.AbstractRepresentation
   :members:
```

The following concrete sub-classes define the representations of existing methods.

```{eval-rst}
.. autoclass:: qunfold.ClassRepresentation

.. autoclass:: qunfold.DistanceRepresentation

.. autoclass:: qunfold.HistogramRepresentation

.. autoclass:: qunfold.EnergyKernelRepresentation

.. autoclass:: qunfold.GaussianKernelRepresentation

.. autoclass:: qunfold.LaplacianKernelRepresentation

.. autoclass:: qunfold.GaussianRFFKernelRepresentation

.. autoclass:: qunfold.OriginalRepresentation
```


## Utilities

The following classes provide functionalities that go beyond the composition of quantification methods.

### Cross-validated training

The `qunfold.sklearn` module allows you to train classification-based quantification methods through cross-validation. Importing this module requires [scikit-learn](https://scikit-learn.org/stable/) to be installed.

```{eval-rst}
.. autoclass:: qunfold.sklearn.CVClassifier
   :members:
   :undoc-members:
   :exclude-members: set_score_request
   :show-inheritance:
```

```{hint}
If you use a bagging classifier (like random forests) with `oob_score=True`, you do not need to use cross-validation. Instead, the quantification method is then trained on the out-of-bag predictions of the bagging classifier.
```

### QuaPy

**Deprecation notice:** The former `qunfold.quapy` module has been moved to [QuaPy](https://github.com/HLT-ISTI/QuaPy). Please consult the [documentation of `quapy.method.composable`](https://hlt-isti.github.io/QuaPy/manuals/methods.html#composable-methods) for integrating qunfold with QuaPy.
