# Developer guide

In the following, we introduce best practices regarding the implementation [workflow](#workflow) before going into detail about how to take out [custom implementations](#custom-implementations).

## Workflow

Before you push to the `main` branch, please test the code and the documentation locally.

### Unit testing

Run tests locally with the `unittest` package.

```bash
python -m venv venv
venv/bin/pip install -e .[tests]
venv/bin/python -m unittest
```

As soon as you push to the `main` branch, GitHub Actions will take out these unit tests, too.


### Documentation

After locally building the documentation, open `docs/build/html/index.html` in your browser.

```bash
. venv/bin/activate
pip install -e .[docs]
cd docs/
sphinx-apidoc --force --output-dir source/ ../qunfold
make html
```

As soon as you push to the `main` branch, GitHub Actions will build the documentation, push it to the `gh-pages` branch, and publish the result on GitHub Pages: [https://mirkobunse.github.io/qunfold](https://mirkobunse.github.io/qunfold)


## Custom implementations

Custom [](#losses) and [](#feature-transformations) can be used in any instance of `GenericMethod`. Use the already existing implementations as examples.


### Losses

The most convenient way of implementing a custom loss is to create a [JAX](https://jax.readthedocs.io/)-powered function `(p, q, M, N) -> loss_value`. From this function, you can create a *FunctionLoss* object to be used in any instance of `GenericMethod`.

```{eval-rst}
.. autoclass:: qunfold.losses.FunctionLoss
```

If you require more freedom in implementing a custom loss, you can also create a sub-class of `AbstractLoss`.

```{eval-rst}
.. autoclass:: qunfold.losses.AbstractLoss
   :members: _instantiate
```


### Feature transformations

To implement a custom feature transformation, you have to create a sub-class of `AbstractTransformer`.

```{eval-rst}
.. autoclass:: qunfold.transformers.AbstractTransformer
   :members:
```
