[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://mirkobunse.github.io/qunfold)
[![CI](https://github.com/mirkobunse/qunfold/actions/workflows/ci.yml/badge.svg)](https://github.com/mirkobunse/qunfold/actions/workflows/ci.yml)


# qunfold | Quantification & Unfolding

This Python package implements our unified framework of algorithms for quantification and unfolding.


## Installation

```
pip install 'qunfold @ git+https://github.com/mirkobunse/qunfold'
```


## Quick start

For detailed information, visit [the documentation](https://mirkobunse.github.io/qunfold).

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
