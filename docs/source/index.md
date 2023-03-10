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
