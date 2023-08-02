import argparse
import itertools
import numpy as np
import os
import pandas as pd
import quapy as qp
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from qunfold import ClassTransformer, GenericMethod, LeastSquaresLoss, BlobelLoss
from qunfold.quapy import QuaPyWrapper
from qunfold.sklearn import CVClassifier
from sklearn.linear_model import LogisticRegression

def condition_number(M, norm="blobel"):
    """Compute the condition number of a matrix"""
    if norm != "blobel": # norm in [None, 1, -1, 2, -2, inf, -inf, "fro"]
        return np.linalg.cond(M, norm)
    s = np.linalg.svd(M, compute_uv=False) # norm="blobel" equals norm="2" and norm="None"
    s = s[s != 0]
    return s[0] / s[-1]

class PseudoInverse(GenericMethod):
    def __init__(self, transformer, normalize=True, **kwargs):
        GenericMethod.__init__(self, None, transformer, **kwargs)
        self.normalize = normalize
    def solve(self, q, M, N=None): # overwrite GenericMethod.solve
        p = np.dot(np.linalg.pinv(M), q) # solve via pseudo-inverse
        if self.normalize:
            p = np.maximum(0, p) # clip values that are smaller than 0
            return p / p.sum() # normalize to 1
        return p # not a probability density

def trial(transformer_config, trn_data, val_gen, seed, n_trials):
    """A single trial of lequa.main()"""
    i_transformer, is_probabilistic, class_weight, C, transformer = transformer_config
    np.random.seed(seed)
    print(
        f"INI [{i_transformer+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"p={is_probabilistic}, w={class_weight}, C={C} starting"
    )

    results = []
    solvers = [ # (solver_name, generic_method)
        ("lsq", GenericMethod(LeastSquaresLoss(), transformer)),
        ("blobel", GenericMethod(BlobelLoss(), transformer)),
        ("pinv-normalize", PseudoInverse(transformer, normalize=True)),
        ("pinv-nonormalize", PseudoInverse(transformer, normalize=False)),
    ]
    for i_solver, (solver_name, generic_method) in enumerate(solvers):
        quapy_method = QuaPyWrapper(generic_method).fit(trn_data)

        # compute all condition numbers
        condition_numbers = {}
        for norm in [ 1, -1, 2, -2, np.inf, -np.inf, "fro" ]:
            condition_numbers["cond_" + str(norm)] = condition_number(
                quapy_method.generic_method.M,
                norm
            )

        # evaluate the method on the validation samples
        errors = {}
        p_true, p_pred = qp.evaluation.prediction(quapy_method, val_gen)
        for error_metric in [ "ae", "rae", "se" ]:
            error_values = qp.error.from_name(error_metric)(p_true, p_pred)
            errors[error_metric] = error_values.mean()
            errors[error_metric + "_std"] = error_values.std()

        solver_dict = {
            "is_probabilistic": is_probabilistic,
            "class_weight": class_weight,
            "C": C,
            "solver": solver_name,
        }
        results.append(solver_dict | condition_numbers | errors)
        print(
            f"VAL [{i_transformer+1:02d}/{n_trials:02d}]:",
            datetime.now().strftime('%H:%M:%S'),
            f"p={is_probabilistic}, w={class_weight}, C={C} validated with {solver_name} ({i_solver+1}/{len(solvers)})"
        )
    return results

def fit_classifier(classifier_config, trn_data, seed, is_test_run, n_classifiers):
    i_classifier, class_weight, C = classifier_config
    np.random.seed(seed)
    classifier = CVClassifier(
        LogisticRegression(
            class_weight = class_weight,
            C = C,
            max_iter = 100 if not is_test_run else 3,
        ),
        n_estimators = 5 if not is_test_run else 3,
        random_state = seed,
    ).fit(*trn_data.Xy)
    transformers = []
    for is_probabilistic in [ True, False ]:
        transformers.append((
            str(is_probabilistic),
            str(class_weight),
            C,
            ClassTransformer(
                classifier,
                is_probabilistic = is_probabilistic,
                fit_classifier = False
            )
        ))
    return transformers

def main(
        output_path,
        seed = 867,
        n_jobs = -1,
        is_test_run = False,
    ):
    if n_jobs < 0:
        n_jobs = None
    print(f"Starting a lequa experiment to produce {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    if len(os.path.dirname(output_path)) > 0: # ensure that the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(seed)
    qp.environ["_R_SEED"] = seed
    qp.environ["SAMPLE_SIZE"] = 1000
    qp.environ["N_JOBS"] = 1

    # load the data; subsample for testing
    trn_data, val_gen, _ = qp.datasets.fetch_lequa2022(task="T1B")
    val_gen.true_prevs.df = val_gen.true_prevs.df[:100] # TODO use all samples
    if is_test_run: # use a minimal testing configuration
        trn_data = trn_data.split_stratified(3000, random_state=seed)[0] # subsample
        val_gen.true_prevs.df = val_gen.true_prevs.df[:3] # use only 3 validation samples


    # fit the classifiers and create transformers
    class_weights = [ None, "balanced" ]
    C_values = [ 1e-3, 1e-2, 1e-1, 1e0, 1e1 ]
    if is_test_run:
        C_values = C_values[:1]
    classifier_configs = [ # (i_classifier, class_weight, C)
        (x[0], x[1][0], x[1][1])
        for x in enumerate(itertools.product(class_weights, C_values))
    ]
    configured_fit_classifier = partial(
        fit_classifier,
        trn_data = trn_data,
        seed = seed,
        is_test_run = is_test_run,
        n_classifiers = len(classifier_configs),
    )
    print(f"Fitting {len(classifier_configs)} classifiers")
    transformer_configs = [] # (is_probabilistic, class_weight, C, transformer)
    with Pool(n_jobs) as pool:
        for x in pool.imap(configured_fit_classifier, classifier_configs):
            transformer_configs.extend(x)
    transformer_configs = [ # (i_transformer, is_probabilistic, class_weight, C, transformer)
        (x[0], x[1][0], x[1][1], x[1][2], x[1][3])
        for x in enumerate(transformer_configs)
    ]
    print(f"Created {len(transformer_configs)} transformers from {len(classifier_configs)} classifiers")

    # parallelize over all transformers
    configured_trial = partial(
        trial,
        trn_data = trn_data,
        val_gen = val_gen,
        seed = seed,
        n_trials = len(transformer_configs),
    )
    print(f"Starting trials for {len(transformer_configs)} transformers")
    results = []
    with Pool(n_jobs) as pool:
        for x in pool.imap(configured_trial, transformer_configs):
            results.extend(x)
    df = pd.DataFrame(results)
    df.to_csv(output_path) # store the results
    print(f"{df.shape[0]} results succesfully stored at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument('--n_jobs', type=int, default=-1, metavar='N',
                        help='number of processes, or the number of cores (default: -1)')
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.seed,
        args.n_jobs,
        args.is_test_run,
    )
