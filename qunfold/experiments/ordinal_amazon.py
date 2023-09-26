import argparse
import itertools
import numpy as np
import os
import pandas as pd
import quapy as qp
from copy import deepcopy
from qunfold import ACC, ClassTransformer, GenericMethod, LeastSquaresLoss, TikhonovRegularized
from qunfold.quapy import QuaPyWrapper
from qunfold.sklearn import CVClassifier
from qunfold.losses import _tikhonov_matrix, _tikhonov
from sklearn.linear_model import LogisticRegression
from time import time

class MultiGridSearch:
    """Grid search for multiple methods, each having their own hyper-parameter grid, and for multiple validation protocols with multiple error metrics."""
    def __init__(self, error_metrics, app_oq_frac, seed=876, n_jobs=1):
        self.error_metrics = error_metrics
        self.app_oq_frac = app_oq_frac
        self.seed = seed
        self.n_jobs = n_jobs
    def validate(self, methods, trn_data, protocols):
        """This validation phase takes out hyper-parameter optimization."""
        trials = []
        for method_name, method, param_grid in methods:
            trials.extend([ # expand param_grid to (method_name, method, params)
                (
                    method_name,
                    QuaPyWrapper(deepcopy(method)),
                    {k: val[i] for i, k in enumerate(list(param_grid.keys()))}
                )
                for val in itertools.product(*list(param_grid.values()))
            ])
        results = qp.util.parallel(
            self._delayed_validate,
            (
                (*trial, trn_data, protocol_name, protocol, i_trial, len(trials)*len(protocols))
                for i_trial, (trial, (protocol_name, protocol))
                in enumerate(itertools.product(trials, protocols.items()))
            ),
            seed = self.seed,
            n_jobs = self.n_jobs,
        )
        self.param_scores = {}
        self.best_scores = {} # (method_name, score_key) -> score
        self.best_stds = {} # (method_name, score_key) -> std of score
        self.best_params = {} # (method_name, score_key) -> params
        self.best_methods = {} # (method_name, score_key) -> method
        for method_name, params, param_scores, method in results:
            self.param_scores[str(params)] = param_scores # dict of dicts
            for protocol_name, protocol in protocols.items():
                for score_key, score in param_scores.items():
                    if not score_key.endswith("_std"):
                        if score < self.best_scores.get((method_name, score_key), np.inf):
                            self.best_scores[(method_name, score_key)] = score
                            self.best_stds[(method_name, score_key)] = param_scores[score_key+"_std"]
                            self.best_params[(method_name, score_key)] = params
                            self.best_methods[(method_name, score_key)] = method
        return self
    def _delayed_validate(self, args):
        method_name, method, params, trn_data, protocol_name, protocol, i_trial, n_trials = args
        method.set_params(**params)
        method.fit(trn_data)
        param_scores = {}
        p_true, p_pred = prediction(method, protocol, "VAL", i_trial, n_trials)
        for error_metric in self.error_metrics:
            errors = compute_errors(error_metric, p_true, p_pred)
            param_scores[protocol_name+"_"+error_metric] = np.mean(errors)
            param_scores[protocol_name+"_"+error_metric+"_std"] = np.std(errors)
            if self.app_oq_frac is not None:
                errors = errors[np.argsort(smoothness(p_true))[:round(self.app_oq_frac * len(p_true))]]
                param_scores[protocol_name+"_oq_"+error_metric] = np.mean(errors)
                param_scores[protocol_name+"_oq_"+error_metric+"_std"] = np.std(errors)
        return method_name, params, param_scores, method
    def test(self, protocols):
        self.results = pd.DataFrame(qp.util.parallel(
            self._delayed_test,
            (
                (
                    method_name,
                    score_key,
                    protocols[score_key.split("_")[0]], # protocol
                    self.best_params[(method_name, score_key)], # params
                    method,
                    i_trial,
                    len(self.best_methods)
                )
                for i_trial, ((method_name, score_key), method)
                in enumerate(self.best_methods.items())
            ),
            seed = self.seed,
            n_jobs = self.n_jobs,
        ))
        return self
    def _delayed_test(self, args):
        method_name, score_key, protocol, params, method, i_trial, n_trials = args
        error_metric = score_key.split("_")[-1]
        is_oq = score_key.split("_")[1] == "oq"
        p_true, p_pred = prediction(method, protocol, "TST", i_trial, n_trials)
        errors = compute_errors(error_metric, p_true, p_pred)
        if is_oq:
            errors = errors[np.argsort(smoothness(p_true))[:round(self.app_oq_frac * len(errors))]]
        return {
            "method": method_name,
            "protocol": score_key.split("_")[0] + ("_oq" if is_oq else ""),
            "error_metric": error_metric,
            "error": np.mean(errors),
            "error_std": np.std(errors),
        }

def prediction(method, protocol, desc, i_trial, n_trials):
    p_true, p_pred = [], []
    t_0 = time()
    for i, (X, p) in enumerate(protocol()):
        p_true.append(p)
        p_pred.append(method.quantify(X))
        if (i+1) == protocol.total():
            print(f"{desc}{i_trial+1:4d}/{n_trials}: DONE")
        elif (i+1) % 10 == 0:
            eta = round((time() - t_0) / (i+1) * protocol.total() - time() + t_0)
            print(f"{desc}{i_trial+1:4d}/{n_trials}:{i+1:4d}/{protocol.total()} (ETA: {eta} s)")
    return np.asarray(p_true), np.asarray(p_pred)

def compute_errors(error_metric, p_true, p_pred):
    if error_metric == "nmd": # vectorized MDPA algorithm [cha2002measuring] for NMD
        prefixsums = np.zeros(p_true.shape[0])
        distances = np.zeros(p_true.shape[0])
        for i in range(p_true.shape[1]):
            prefixsums += p_true[:,i] - p_pred[:,i]
            distances += np.abs(prefixsums)
        return distances / (np.sum(p_true, axis=1) * (p_true.shape[1] - 1))
    else:
        raise NotImplementedError()

def smoothness(p_true):
    n_classes = p_true.shape[1]
    T = _tikhonov_matrix(n_classes)
    return np.array([_tikhonov(p, T) for p in p_true])

# custom subclass of SamplesFromDir that does implement self.total
from quapy.data._lequa2022 import SamplesFromDir
class MySamplesFromDir(SamplesFromDir):
    def total(self):
        return len(self.true_prevs.df)

def fetch_amazon(data_path):
    trn_data = qp.data.LabelledCollection.load(
        os.path.join(data_path, "training_data.txt"),
        loader_func = load_amazon_file
    )
    val_gen_app = MySamplesFromDir(
        os.path.join(data_path, "app", "dev_samples"),
        os.path.join(data_path, "app", "dev_prevalences.txt"),
        load_fn = load_amazon_file
    )
    tst_gen_app = MySamplesFromDir(
        os.path.join(data_path, "app", "test_samples"),
        os.path.join(data_path, "app", "test_prevalences.txt"),
        load_fn = load_amazon_file
    )
    val_gen_real = MySamplesFromDir(
        os.path.join(data_path, "real", "dev_samples"),
        os.path.join(data_path, "real", "dev_prevalences.txt"),
        load_fn = load_amazon_file
    )
    tst_gen_real = MySamplesFromDir(
        os.path.join(data_path, "real", "test_samples"),
        os.path.join(data_path, "real", "test_prevalences.txt"),
        load_fn = load_amazon_file
    )
    return trn_data, val_gen_app, tst_gen_app, val_gen_real, tst_gen_real

def load_amazon_file(path):
    D = pd.read_csv(path, sep="\s+").to_numpy(dtype=float)
    return D[:, 1:], D[:, 0].astype(int).flatten() # X, y

def main(
        output_path,
        data_path = "/mnt/data/amazon-oq-bk/roberta",
        seed = 867,
        n_jobs = 1,
        is_test_run = False,
    ):
    print(f"Starting an ordinal_amazon experiment to produce {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    if len(os.path.dirname(output_path)) > 0: # ensure that the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(seed)
    qp.environ["_R_SEED"] = seed
    qp.environ["SAMPLE_SIZE"] = 1000
    qp.environ["N_JOBS"] = 1

    # configure the quantification methods
    clf = CVClassifier(
        LogisticRegression(),
        n_estimators = 10,
        random_state = seed,
    )
    clf_grid = {
        "transformer__classifier__estimator__C": [1e-3, 1e-2, 1e-1, 1e0, 1e1],
        "transformer__classifier__estimator__class_weight": ["balanced", None],
    }
    methods = [ # (method_name, method, param_grid)
        ("ACC", ACC(clf, seed=seed), clf_grid),
        ("o-ACC",
            GenericMethod(
                TikhonovRegularized(LeastSquaresLoss(), 0.01),
                ClassTransformer(clf),
                seed = seed,
            ),
            dict(clf_grid, loss__weights=[(1, 1e-1), (1, 1e-3), (1, 1e-5)]) # extend
        ),
    ]

    # load the data
    trn_data, val_gen_app, tst_gen_app, val_gen_real, tst_gen_real = fetch_amazon(data_path)

    if is_test_run: # use a minimal testing configuration
        clf.set_params(n_estimators = 3, estimator__max_iter = 3)
        clf_grid = {
            "transformer__classifier__estimator__C": [1e1],
        }
        methods = [ # (method_name, method, param_grid)
            ("ACC", ACC(clf, seed=seed), clf_grid),
            ("o-ACC",
                GenericMethod(
                    TikhonovRegularized(LeastSquaresLoss(), None),
                    ClassTransformer(clf),
                    seed = seed,
                ),
                dict(clf_grid, loss__weights=[(1, 1e-1), (1, 1e-3)]) # extend
            ),
        ]
        trn_data = trn_data.split_stratified(3000, random_state=seed)[0] # subsample
        val_gen_app.true_prevs.df = val_gen_app.true_prevs.df[:10] # only use 10 samples
        tst_gen_app.true_prevs.df = tst_gen_app.true_prevs.df[:10]
        val_gen_real.true_prevs.df = val_gen_real.true_prevs.df[:10]
        tst_gen_real.true_prevs.df = tst_gen_real.true_prevs.df[:10]

    # validation for all methods, all protocols and all error_metrics, then testing
    gs = MultiGridSearch(
        error_metrics = ["nmd"],
        app_oq_frac = .5,
        seed = seed,
        n_jobs = n_jobs,
    ).validate( # TODO store validation results before testing (and recover from there)
        methods,
        trn_data,
        {"app": val_gen_app, "real": val_gen_real}, # protocols
    ).test(
        {"app": tst_gen_app, "real": tst_gen_real},
    )
    gs.results.to_csv(output_path) # store the results
    print(f"{gs.results.shape[0]} results succesfully stored at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str, help="path of an output *.csv file")
    parser.add_argument("--data_path", type=str,
                        default="/mnt/data/amazon-oq-bk/roberta",
                        help="directory of data files")
    parser.add_argument("--seed", type=int, default=876, metavar="N",
                        help="random number generator seed (default: 876)")
    parser.add_argument("--n_jobs", type=int, default=1, metavar="N",
                        help="number of parallel jobs, -1 for all cores (default: 1)")
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.data_path,
        args.seed,
        args.n_jobs,
        args.is_test_run,
    )
