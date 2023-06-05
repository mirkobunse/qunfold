import argparse
import numpy as np
import os
import pandas as pd
import quapy as qp
from functools import partial
from multiprocessing import Pool
from qunfold import ACC, PACC
from qunfold.quapy import QuaPyWrapper
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

def trial(trial_config, trn_data, val_gen, tst_gen, seed, error_metric, n_methods):
    """A single trial of lequa.main()"""
    i_method, (method_name, package, method, param_grid) = trial_config # unpack the tuple
    np.random.seed(seed)
    print(
        f"INI {error_metric} [{i_method+1:02d}/{n_methods:02d}]:",
        f"{package} {method_name} starting"
    )

    # configure and train the method; select the best hyper-parameters
    quapy_method = qp.model_selection.GridSearchQ(
        model = method,
        param_grid = param_grid,
        protocol = val_gen,
        error = error_metric,
        refit = False,
        # verbose = True,
    ).fit(trn_data)
    parameters = quapy_method.best_params_
    quapy_method = quapy_method.best_model_
    print(
        f"VAL {error_metric} [{i_method+1:02d}/{n_methods:02d}]:",
        f"{package} {method_name} selects {parameters}"
    )

    # evaluate the method on the test samples and return the result
    error = qp.evaluation.evaluate(
        quapy_method,
        protocol = tst_gen,
        error_metric = error_metric
    )
    print(
        f"TST {error_metric} [{i_method+1:02d}/{n_methods:02d}]:",
        f"{package} {method_name} yields {error_metric}={error}"
    )
    return {
        "method": method_name,
        "package": package,
        "error_metric": error_metric,
        "error": error,
        "parameters": str(parameters),
    }

def main(
        output_path,
        seed = 867,
        is_test_run = False,
    ):
    print(f"Starting a lequa experiment to produce {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    if len(os.path.dirname(output_path)) > 0: # ensure that the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(seed)
    qp.environ["SAMPLE_SIZE"] = 1000
    qp.environ["N_JOBS"] = 1

    # configure the quantification methods
    clf = BaggingClassifier(
        LogisticRegression(n_jobs=1),
        n_estimators = 100,
        n_jobs = 1,
        random_state = seed,
        oob_score = True,
    )
    clf_grid = {
        "transformer__classifier__estimator__C": [1e-3, 1e-2, 1e-1, 1e0, 1e1],
    }
    qp_clf = clf.estimator
    qp_clf_grid = {
        "classifier__estimator__C": clf_grid["transformer__classifier__estimator__C"],
    }
    methods = [ # (method_name, package, method, param_grid)
        ("ACC", "qunfold", QuaPyWrapper(ACC(clf)), clf_grid),
        ("PACC", "qunfold", QuaPyWrapper(PACC(clf)), clf_grid),
        ("ACC", "QuaPy", qp.method.aggregative.ACC(qp_clf, val_split=100), qp_clf_grid),
    ]

    # load the data
    trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")

    if is_test_run: # use a minimal testing configuration
        clf = BaggingClassifier(
            LogisticRegression(n_jobs=1, max_iter=3),
            n_estimators = 100,
            n_jobs = 1,
            random_state = seed,
            oob_score = True,
        )
        clf_grid = {
            "transformer__classifier__estimator__C": [1e1],
        }
        qp_clf = clf.estimator
        qp_clf_grid = {
            "classifier__C": clf_grid["transformer__classifier__estimator__C"],
        }
        methods = [ # (method_name, package, method, param_grid)
            ("ACC", "qunfold", QuaPyWrapper(ACC(clf)), clf_grid),
            ("ACC", "QuaPy", qp.method.aggregative.ACC(qp_clf, val_split=10), qp_clf_grid),
        ]
        trn_data = trn_data.split_random(2000)[0] # use only 2k training items
        val_gen.true_prevs.df = val_gen.true_prevs.df[:5] # use only 5 validation samples
        tst_gen.true_prevs.df = tst_gen.true_prevs.df[:5] # use only 5 testing samples

    # parallelize over all methods
    results = []
    for error_metric in ['mae', 'mrae']:
        with Pool() as pool:
            trial_metric = partial(
                trial,
                trn_data = trn_data,
                val_gen = val_gen,
                tst_gen = tst_gen,
                seed = seed,
                error_metric = error_metric,
                n_methods = len(methods),
            )
            trial_results = pool.imap(trial_metric, enumerate(methods))
            results.extend(trial_results)
    df = pd.DataFrame(results)
    df.to_csv(output_path) # store the results
    print(f"{df.shape[0]} results succesfully stored at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.seed,
        args.is_test_run,
    )
