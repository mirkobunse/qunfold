import argparse
import numpy as np
import os
import pandas as pd
import quapy as qp
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from qunfold import ACC, PACC, HDy, EDy, RUN, ClassTransformer
from qunfold.quapy import QuaPyWrapper
from qunfold.sklearn import CVClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from time import time
from tqdm.auto import tqdm

def trial(trial_config, trn_data, tst_gen, seed, n_trials):
    """A single trial of lequa.main()"""
    i_method, method_name, package, method, gtol = trial_config
    np.random.seed(seed)
    print(
        f"INI [{i_method+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{package} {method_name} starting"
    )

    # train the method with fixed hyper-parameters
    quapy_method = method.fit(trn_data)
    print(
        f"Skipping validation of {package} {method_name} due to fixed hyper-parameters."
    )

    # evaluate the method on the test samples and return the result
    t_0 = time()
    p_true, p_pred = qp.evaluation.prediction(quapy_method, protocol=tst_gen)
    prediction_time = (time() - t_0) / len(p_true) # average prediction_time
    p_ae = qp.error.ae(p_true, p_pred)
    ae = p_ae.mean()
    ae_std = p_ae.std()
    p_rae = qp.error.rae(p_true, p_pred)
    rae = p_rae.mean()
    rae_std = p_rae.std()
    print(
        f"TST [{i_method+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{package} {method_name} tested",
        f"ae={ae:.4f}+-{ae_std:.4f}, rae={rae:.4f}+-{rae_std:.4f}",
    )
    return {
        "method": method_name,
        "package": package,
        "ae": ae,
        "ae_std": ae_std,
        "rae": rae,
        "rae_std": rae_std,
        "prediction_time": prediction_time,
        "gtol": gtol,
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
    qp.environ["_R_SEED"] = seed
    qp.environ["SAMPLE_SIZE"] = 1000
    qp.environ["N_JOBS"] = 1

    # configure the quantification methods
    gtols = [ 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 ]
    clf = lambda C: CVClassifier(
        LogisticRegression(C=C),
        n_estimators = 5,
        random_state = seed,
    )
    if is_test_run:
        gtols = [ 1e-8, 1e-4, 1e-1 ]
        clf = lambda C: CVClassifier(
            LogisticRegression(C=C, max_iter=3),
            n_estimators = 3,
            random_state = seed,
        )
    methods = [ # (method_name, package, method, gtol)
        ("ACC", "QuaPy", qp.method.aggregative.ACC(clf(1.0).estimator, val_split=5), -1),
        ("PACC", "QuaPy", qp.method.aggregative.PACC(clf(10.).estimator, val_split=5), -1),
        # ("HDy", "QuaPy",
        #     qp.method.aggregative.DistributionMatching(
        #         classifier = clf(.1).estimator,
        #         divergence = 'HD',
        #         cdf = False,
        #         nbins = 4,
        #     ), -1),
    ]
    for gtol in gtols:
        method_args = {
            "seed": seed,
            "solver_options": {"gtol": gtol, "maxiter": 1000},
        }
        methods.extend([
            ("ACC_ae", "qunfold", QuaPyWrapper(ACC(clf(.1), **method_args)), gtol),
            ("ACC_rae", "qunfold", QuaPyWrapper(ACC(clf(10.), **method_args)), gtol),
            ("PACC", "qunfold", QuaPyWrapper(PACC(clf(.1), **method_args)), gtol),
            # ("HDy", "qunfold", QuaPyWrapper(HDy(clf(.1), 6, **method_args)), gtol),
        ])

    # load the data
    trn_data, _, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")

    # for now, only a subset of the samples is used; TODO use all samples
    tst_gen.true_prevs.df = tst_gen.true_prevs.df[:500]

    if is_test_run: # use a minimal testing configuration
        trn_data = trn_data.split_stratified(3000, random_state=seed)[0] # subsample
        tst_gen.true_prevs.df = tst_gen.true_prevs.df[:3] # use only 3 testing samples

    # parallelize over all methods
    configured_trial = partial(
        trial,
        trn_data = trn_data,
        tst_gen = tst_gen,
        seed = seed,
        n_trials = len(methods),
    )
    trials = [ # (i_method, method_name, package, method, gtol)
        (x[0], x[1][0], x[1][1], x[1][2], x[1][3])
        for x in enumerate(methods)
    ]
    print(f"Starting {len(trials)} trials")
    results = []
    with Pool() as pool:
        results.extend(pool.imap(configured_trial, trials))
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
