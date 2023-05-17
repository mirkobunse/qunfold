```{toctree}
:hidden:

self
api
developer-guide
```

# Quickstart

This Python package implements our unified framework of algorithms for quantification and unfolding.


## Installation

```
pip install 'qunfold @ git+https://github.com/mirkobunse/qunfold'
```

Moreover, you will need a [JAX](https://jax.readthedocs.io/) backend. Typically, the CPU backend will be ideal:

```
pip install "jax[cpu]"
```

**Updating:** To update an existing installation of `qunfold`, run

```
pip install --force-reinstall --no-deps 'qunfold @ git+https://github.com/mirkobunse/qunfold@main'
```

**Troubleshooting:** Starting from `pip 23.1.2`, you have to install `setuptools` and `wheel` explicitly. If you receive a "NameError: name 'setuptools' is not defined", you need to execute the following command before installing `qunfold`.

```
pip install --upgrade pip setuptools wheel
```


## Usage

Basically, you use this package as follows:

```python
from qunfold import ACC
from sklearn.ensemble import RandomForestClassifier

acc = ACC( # a scikit-learn bagging classifier with oob_score is needed
    RandomForestClassifier(oob_score=True)
)

# X_trn, y_trn = my_training_data(...)
acc.fit(X_trn, y_trn)

# X_tst = my_testing_data(...)
p_est = acc.predict(X_tst) # return a prevalence vector
```
