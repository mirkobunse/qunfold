# Experiments

We validate the performance of `qunfold`, in comparison to [QuaPy](https://github.com/HLT-ISTI/QuaPy), through experiments with the [LeQua2022 data](LeQua2022.github.io/). We document these experiments for transparency and to facilitate implementations of similar experiments.


## Setup

You can install the dependencies either in an isolated [Docker](https://docs.docker.com/) environment or locally with `venv`.

### Docker setup (optional)

We provide an isolated Docker environment for conveniently running the experiments. To create the image and start a container from it, call

```bash
cd docker/
make
./run.sh
```

Inside the container, navigate to the `qunfold` repository and install the dependencies

```bash
cd /mnt/home/.../qunfold/
pip install .[experiments]
```

### Local setup (alternative)

Without Docker, use `venv` to install the dependencies

```bash
python -m venv venv
venv/bin/pip install -e .[experiments]
```


## Running the experiments

The experiments are implemented through a main function that you can call as follows:

```bash
venv/bin/python -m qunfold.experiments.lequa lequa_results.csv
```

With the `--is_test_run` switch, you can execute the entire code path with minimal effort, to test whether the experiment is working. This functionality is particularly helpful if you make changes to the experiments.

Finally, the tables with average performance values are created as follows:

```bash
venv/bin/python -m qunfold.experiments.create_table lequa_results.csv lequa_results.tex
```
