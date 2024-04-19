import argparse
import itertools
import numpy as np
import os
import pandas as pd
import quapy as qp
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from qunfold import ACC, PACC, HDy, EDy, RUN, KMM, ClassTransformer, GaussianRFFKernelTransformer, LeastSquaresLoss, EnergyKernelTransformer, LinearMethod, KDEyHD, KDEyCS, KDEyMLQP
from qunfold.quapy import QuaPyWrapper
from qunfold.sklearn import CVClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from time import time
from tqdm.auto import tqdm
from quapy.method._kdey import KDEyML as KDEyML_QuaPy
import warnings
warnings.filterwarnings("ignore")

def trial(trial_config, trn_data, val_gen, tst_gen, seed, n_trials):
    """A single trial of lequa.main()"""
    i_method, method_name, package, method, param_grid, error_metric = trial_config
    np.random.seed(seed)
    print(
        f"INI [{i_method+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{package} {method_name} / {error_metric} starting"
    )

    # configure and train the method; select the best hyper-parameters
    if param_grid is not None:
        quapy_method = qp.model_selection.GridSearchQ(
            model = method,
            param_grid = param_grid,
            protocol = val_gen,
            error = "m" + error_metric, # ae -> mae, rae -> mrae
            refit = False,
            raise_errors = True,
            verbose = True,
        ).fit(trn_data)
        parameters = quapy_method.best_params_
        val_error = quapy_method.best_score_
        quapy_method = quapy_method.best_model_
        print(
            f"VAL [{i_method+1:02d}/{n_trials:02d}]:",
            datetime.now().strftime('%H:%M:%S'),
            f"{package} {method_name} validated {error_metric}={val_error:.4f} {parameters}"
        )
    else:
        quapy_method = method.fit(trn_data)
        parameters = None
        val_error = -1
        print(
            f"Skipping validation of {package} {method_name} due to fixed hyper-parameters."
        )

    # evaluate the method on the test samples and return the result
    t_0 = time()
    errors = qp.evaluation.evaluate( # errors of all predictions
        quapy_method,
        protocol = tst_gen,
        error_metric = error_metric
    )
    prediction_time = (time() - t_0) / len(errors) # average prediction_time
    error = errors.mean()
    error_std = errors.std()
    print(
        f"TST [{i_method+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{package} {method_name} tested {error_metric}={error:.4f}+-{error_std:.4f}"
    )
    return {
        "method": method_name,
        "package": package,
        "error_metric": error_metric,
        "error": error,
        "error_std": error_std,
        "prediction_time": prediction_time,
        "val_error": val_error,
        "parameters": str(parameters),
    }

def main(
        output_path,
        n_jobs = 1,
        seed = 867,
        is_full_run = False,
        is_test_run = False,
    ):
    print(f"Starting a lequa experiment to produce {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    if len(os.path.dirname(output_path)) > 0: # ensure that the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(seed)
    qp.environ["_R_SEED"] = seed
    qp.environ["SAMPLE_SIZE"] = 1000

    # configure the quantification methods
    clf = CVClassifier(
        LogisticRegression(),
        n_estimators = 5,
        random_state = seed,
    )
    clf_grid = {
        "transformer__classifier__estimator__C": [1e-3, 1e-2, 1e-1, 1e0, 1e1],
    }
    qp_clf = clf.estimator
    qp_clf_grid = {
        "classifier__C": clf_grid["transformer__classifier__estimator__C"],
    }
    methods = [ # (method_name, package, method, param_grid)
        #("ACC", "qunfold", QuaPyWrapper(ACC(clf, seed=seed)), clf_grid),
        #("ACC", "QuaPy", qp.method.aggregative.ACC(qp_clf, val_split=5), qp_clf_grid),
        #("PACC", "qunfold", QuaPyWrapper(PACC(clf, seed=seed)), clf_grid),
        #("PACC", "QuaPy", qp.method.aggregative.PACC(qp_clf, val_split=5), qp_clf_grid),
        #("HDy", "qunfold",
        #    QuaPyWrapper(HDy(clf, 2, seed=seed)),
        #    {
        #        "transformer__preprocessor__classifier__estimator__C":
        #            clf_grid["transformer__classifier__estimator__C"],
        #        "transformer__n_bins": [2, 4, 6],
        #    }
        #),
        #("HDy", "QuaPy",
        #    qp.method.aggregative.DistributionMatching(
        #        classifier = qp_clf,
        #        divergence = 'HD',
        #        cdf = False
        #    ),
        #    dict(qp_clf_grid, nbins = [2, 4, 6]) # extend the qp_clf_grid
        #),
        #("EDy", "qunfold",
        #    QuaPyWrapper(EDy(clf, seed=seed)),
        #    {
        #        "transformer__preprocessor__classifier__estimator__C":
        #            clf_grid["transformer__classifier__estimator__C"],
        #    }
        #),
        #("RUN", "qunfold", QuaPyWrapper(RUN(ClassTransformer(clf), seed=seed)), clf_grid),
        #("KMMe", "qunfold", QuaPyWrapper(KMM(kernel="energy", seed=seed)), None),
        #("KMMey", "qunfold", # KMM with the energy kernel after classification
        #    QuaPyWrapper(LinearMethod(
        #        LeastSquaresLoss(),
        #        EnergyKernelTransformer(preprocessor=ClassTransformer(
        #            clf,
        #            is_probabilistic = True,
        #        )),
        #        seed = seed
        #    )),
        #    {
        #        "transformer__preprocessor__classifier__estimator__C":
        #            clf_grid["transformer__classifier__estimator__C"],
        #    }
        #),
        #("KMMr", "qunfold",
        #    QuaPyWrapper(KMM(kernel="rff", seed=seed)),
        #    { "transformer__sigma": [1e-2, 1e-1, 1e0, 1e1, 1e2] }
        #),
        #("KMMry", "qunfold", # KMM with the Gaussian RFF kernel after classification
        #    QuaPyWrapper(LinearMethod(
        #        LeastSquaresLoss(),
        #        GaussianRFFKernelTransformer(
        #            seed = seed,
        #            preprocessor = ClassTransformer(
        #                clf,
        #                is_probabilistic = True,
        #            ),
        #        ),
        #        seed = seed
        #    )),
        #    {
        #        "transformer__sigma": [1e-2, 1e-1, 1e0, 1e1, 1e2],
        #        "transformer__preprocessor__classifier__estimator__C":
        #            clf_grid["transformer__classifier__estimator__C"],
        #    }
        #),
        # ("KDEyML", "QuaPy", qp.method.aggregative.KDEyML(qp_clf, val_split=5),
        #     {
        #         "bandwidth" : [1e-2, 1e-1, 1e0, 1e1, 1e2],
        #         "classifier__C" : [1e-3, 1e-2, 1e-1, 1e0, 1e1]
        #     }
        # ),
        ("SLD", "QuaPy", qp.method.aggregative.EMQ(qp_clf),
            {
                "classifier__C" : np.logspace(-3, 3, 7),
                "classifier__class_weight" : ['balanced', None],
            }
        ),
        #("KDEyHD", "QuaPy", qp.method.aggregative.KDEyHD(qp_clf, val_split=5),
        #    {
        #        "bandwidth" : [1e-2, 1e-1, 1e0, 1e1, 1e2],
        #        "classifier__C" : [1e-3, 1e-2, 1e-1, 1e0, 1e1]
        #    }
        #),
        #("KDEyCS", "QuaPy", qp.method.aggregative.KDEyCS(qp_clf, val_split=5),
        #    {
        #        "bandwidth" : [1e-2, 1e-1, 1e0, 1e1, 1e2],
        #        "classifier__C" : [1e-3, 1e-2, 1e-1, 1e0, 1e1]
        #    }
        #),
        #("KDEyHD", "qunfold", QuaPyWrapper(KDEyHD(clf, bandwidth=0.1, random_state=seed)), 
        #    {
        #        "transformer__bandwidth" : [1e-2, 1e-1, 1e0, 1e1, 1e2],
        #        **clf_grid,
        #    }
        #),
        #("KDEyML", "qunfold", QuaPyWrapper(KDEyML(clf, bandwidth=0.1)), 
        #    {
        #        #"bandwidth" : [1e-2, 1e-1, 1e0, 'scott', 'silverman'],
        #        "bandwidth" : ['scott', 'silverman'],
        #        #"classifier__estimator__C": clf_grid["transformer__classifier__estimator__C"],
        #        "classifier__estimator__C": [0.01, 0.1, 0.5, 1.0, 10.],
        #    }
        #),
        ("KDEyMLQP", "qunfold", QuaPyWrapper(KDEyMLQP(clf, bandwidth=0.1, random_state=seed)), 
            {
                "bandwidth" : np.linspace(0.01, 0.2, 20),
                "classifier__estimator__C": np.logspace(-3, 3, 7),
                "classifier__estimator__class_weight" : ['balanced', None],
            }
        ),
        ("KDEyML", "QuaPy", KDEyML_QuaPy(qp_clf, random_state=seed),
            {
                "bandwidth" : np.linspace(0.01, 0.2, 20),
                "classifier__C": np.logspace(-3, 3, 7),
                "classifier__class_weight" : ['balanced', None],
            }
        ),
        #("KDEyCS", "qunfold", QuaPyWrapper(KDEyCS(clf, bandwidth=0.1)), 
        #    {
        #        "transformer__bandwidth" : [1e-2, 1e-1, 1e0, 1e1, 1e2],
        #        **clf_grid,
        #    }
        #),
    ]

    # load the data
    trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")

    if is_test_run: # use a minimal testing configuration
        clf.set_params(n_estimators = 3, estimator__max_iter = 3)
        clf_grid = {
            "transformer__classifier__estimator__C": [1e1],
        }
        qp_clf = clf.estimator
        qp_clf_grid = {
            "classifier__C": clf_grid["transformer__classifier__estimator__C"],
        }
        methods = [ # (method_name, package, method, param_grid)
            #("ACC", "qunfold", QuaPyWrapper(ACC(clf, seed=seed)), clf_grid),
            #("ACC", "QuaPy", qp.method.aggregative.ACC(qp_clf, val_split=3), qp_clf_grid),
            #("PACC", "qunfold", QuaPyWrapper(PACC(clf, seed=seed)), clf_grid),
            #("PACC", "QuaPy", qp.method.aggregative.PACC(qp_clf, val_split=3), qp_clf_grid),
            #("HDy", "qunfold", QuaPyWrapper(HDy(clf, 2, seed=seed)), None),
            #("HDy", "QuaPy",
            #    qp.method.aggregative.DistributionMatching(
            #        classifier = qp_clf,
            #        divergence = 'HD',
            #        cdf = False
            #    ),
            #    None
            #),
            #("EDy", "qunfold", QuaPyWrapper(EDy(clf, seed=seed)), None),
            #("RUN", "qunfold", QuaPyWrapper(RUN(ClassTransformer(clf), seed=seed)), None),
            #("KMMe", "qunfold", QuaPyWrapper(KMM(kernel="energy", seed=seed)), None),
            #("KMMey", "qunfold", # KMM with the energy kernel after classification
            #    QuaPyWrapper(LinearMethod(
            #        LeastSquaresLoss(),
            #        EnergyKernelTransformer(preprocessor=ClassTransformer(
            #            clf,
            #            is_probabilistic = True,
            #        )),
            #        seed = seed
            #    )),
            #    {
            #        "transformer__preprocessor__classifier__estimator__C":
            #            clf_grid["transformer__classifier__estimator__C"],
            #    }
            #),
            #("KMMr", "qunfold",
            #    QuaPyWrapper(KMM(kernel="rff", seed=seed)),
            #    { "transformer__sigma": [ 1e-1 ] }
            #),
            #("KMMry", "qunfold", # KMM with the Gaussian RFF kernel after classification
            #    QuaPyWrapper(LinearMethod(
            #        LeastSquaresLoss(),
            #        GaussianRFFKernelTransformer(
            #            seed = seed,
            #            preprocessor = ClassTransformer(
            #                clf,
            #                is_probabilistic = True,
            #            ),
            #        ),
            #        seed = seed
            #    )),
            #    {
            #        "transformer__sigma": [ 1e-1 ],
            #        "transformer__preprocessor__classifier__estimator__C":
            #            clf_grid["transformer__classifier__estimator__C"],
            #    }
            #),
            #("KDEyHD", "qunfold", QuaPyWrapper(KDEyHD(clf, bandwidth=0.1, random_state=seed)), 
            #{
            #    "transformer__bandwidth" : [1e-1],
            #    **clf_grid,
            #}
            #),
            ("KDEyMLQP", "qunfold", QuaPyWrapper(KDEyMLQP(qp_clf, random_state=seed)), 
                {
                    "bandwidth" : np.linspace(0.01, 0.2, 20),
                    "classifier__estimator__C": np.logspace(-3, 3, 7),
                    "classifier__estimator__class_weight" : ['balanced', None],
                }
            ),
            ("KDEyML", "QuaPy", KDEyML_QuaPy(qp_clf, random_state=seed),
                {
                    "bandwidth" : np.linspace(0.01, 0.2, 20),
                    "classifier__C": np.logspace(-3, 3, 7),
                    "classifier__class_weight" : ['balanced', None],
                }
            ),
            ("SLD", "QuaPy", qp.method.aggregative.EMQ(qp_clf),
                {
                    "classifier__C" : [1e-2, 1e-1]
                }
            ),
            #("KDEyCS", "qunfold", QuaPyWrapper(KDEyCS(clf, bandwidth=0.1)), 
            #    {
            #        "transformer__bandwidth" : [1e-1],
            #        **clf_grid,
            #    }
            #),
        ]
        trn_data = trn_data.split_stratified(3000, random_state=seed)[0] # subsample
        val_gen.true_prevs.df = val_gen.true_prevs.df[:3] # use only 3 validation samples
        tst_gen.true_prevs.df = tst_gen.true_prevs.df[:3] # use only 3 testing samples
    elif not is_full_run:
        val_gen.true_prevs.df = val_gen.true_prevs.df[:100] # use only 100 validation samples
        tst_gen.true_prevs.df = tst_gen.true_prevs.df[:500] # use only 500 testing samples

    # parallelize over all methods
    error_metrics = ['ae', 'rae']
    configured_trial = partial(
        trial,
        trn_data = trn_data,
        val_gen = val_gen,
        tst_gen = tst_gen,
        seed = seed,
        n_trials = len(methods) * len(error_metrics),
    )
    trials = [ # (i_method, method_name, package, method, param_grid, error_metric)
        (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][1])
        for x in enumerate(itertools.product(methods, error_metrics))
    ]
    print(
        f"Starting {len(trials)} trials",
        f"with {len(val_gen.true_prevs.df)} validation",
        f"and {len(tst_gen.true_prevs.df)} testing samples"
    )
    results = []
    with Pool(n_jobs if n_jobs > 0 else None) as pool:
        results.extend(pool.imap(configured_trial, trials))
    df = pd.DataFrame(results)
    df.to_csv(output_path) # store the results
    print(f"{df.shape[0]} results succesfully stored at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('--n_jobs', type=int, default=1, metavar='N',
                        help='number of concurrent jobs or 0 for all processors (default: 1)')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument("--is_full_run", action="store_true",
                        help="whether to use all 1000 validation and 5000 testing samples")
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.n_jobs,
        args.seed,
        args.is_full_run,
        args.is_test_run,
    )
