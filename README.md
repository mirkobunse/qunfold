# qunfold | Quantification & Unfolding

This Python package implements our unified framework of algorithms for quantification and unfolding.


## Installation

```
pip install 'qunfold @ git+https://github.com/mirkobunse/qunfold'
```


## Quick start

```python
from qunfold import ACC
from sklearn.ensemble import RandomForestClassifier

acc = ACC( # a scikit-learn bagging classifier with oob_score is needed
    RandomForestClassifier(oob_score=true)
)

# X_trn, y_trn = my_training_data(...)
acc.fit(X_trn, y_trn)

# X_tst = my_testing_data(...)
p_est = acc.predict(X_tst) # return a prevalence vector
```


## Development / unit testing

Run tests locally with the `unittest` package.

```
python -m venv venv
venv/bin/pip install .[tests]
venv/bin/python -m unittest
```
