[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://mirkobunse.github.io/qunfold)
[![CI](https://github.com/mirkobunse/qunfold/actions/workflows/ci.yml/badge.svg)](https://github.com/mirkobunse/qunfold/actions/workflows/ci.yml)


# qunfold | Quantification & Unfolding

This Python package implements composable methods for quantification and unfolding.


## Installation

```
pip install 'qunfold @ git+https://github.com/mirkobunse/qunfold'
```

Moreover, you will need a [JAX](https://jax.readthedocs.io/) backend. Typically, the CPU backend will be ideal:

```
pip install "jax[cpu]"
```


## Quick start

For detailed information, visit [the documentation](https://mirkobunse.github.io/qunfold).

Basically, you use this package as follows:

```python
from qunfold import ACC # Adjusted Classify and Count
from sklearn.ensemble import RandomForestClassifier

acc = ACC( # use OOB predictions for training the quantifier
    RandomForestClassifier(oob_score=True)
)
acc.fit(X_trn, y_trn) # fit to training data
p_hat = acc.predict(X_tst) # estimate a prevalence vector
```
