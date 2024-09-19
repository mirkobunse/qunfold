```{toctree}
:hidden:

self
api
developer-guide
experiments
```

# Quickstart

The Python package [qunfold](https://github.com/mirkobunse/qunfold) implements our unified framework of algorithms for quantification and unfolding. It is designed for enabling the composition of novel methods from existing and easily customized loss functions and data representations. Moreover, this package leverages a powerful optimization back-end to yield state-of-the-art performances for all compositions.


## Installation

```
pip install --upgrade pip setuptools wheel
pip install 'qunfold @ git+https://github.com/mirkobunse/qunfold'
```

Moreover, you will need a [JAX](https://jax.readthedocs.io/) backend. Typically, the CPU backend will be ideal:

```
pip install "jax[cpu]"
```

### Upgrading

To upgrade an existing installation of `qunfold`, run

```
pip install --force-reinstall --no-deps 'qunfold @ git+https://github.com/mirkobunse/qunfold@main'
```


## Usage

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

You can easily compose new quantification methods from existing loss functions and data representations. In the following example, we compose the ordinal variant of ACC and prepare it for being used in [QuaPy](https://github.com/HLT-ISTI/QuaPy).

```python
# the ACC loss, regularized with strength 0.01 for ordinal quantification
loss = TikhonovRegularized(LeastSquaresLoss(), 0.01)

# the original data representation of ACC with 10-fold cross-validation
representation = ClassRepresentation(CVClassifier(LogisticRegression(), 10))

# the ordinal variant of ACC, ready for being used in QuaPy
ordinal_acc = QuaPyWrapper(LinearMethod(loss, representation))
```
