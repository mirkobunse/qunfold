import concurrent.futures as cf
import numpy as np
import quapy as qp
import time
from sklearn.linear_model import LogisticRegression
from qunfold import PACC
from qunfold.quapy import QuaPyWrapper
from qunfold.sklearn import CVClassifier

#
# A parallel prediction, where the process coordinator reads all files, completes
# in ~ 35 % of the original runtime with 8 cores.
#
#     evaluate(m, protocol=tst_gen, error_metric="mae")
#
# If the reading of files is defered to the workers, the prediction is not faster
# than the speed-up that is already achieved. Hence, I have removed the implementation
# of defered reading to keep this code simple.
#
# I could not produce a similar speed-up for qp.model_selection.GridSearchQ, likely
# because this class already paralellizes its workload over the hyper-parameters.
# I could not even observe a speed-up when I parallelized jointly over hyper-parameters
# and sample predictions.
#

def prediction(model, protocol, n_jobs=None):
    """Parallel variant of quapy.evaluation.prediction"""
    estim_prevs, true_prevs = [], []
    with cf.ProcessPoolExecutor(max_workers=n_jobs) as p:
        futures = { p.submit(_prediction_worker, model, sample) for sample in protocol() }
        for future in cf.as_completed(futures):
            result = future.result()
            estim_prevs.append(result[0])
            true_prevs.append(result[1])
    return np.asarray(true_prevs), np.asarray(estim_prevs)

def _prediction_worker(model, sample):
    return model.quantify(sample[0]), sample[1] # sample = (sample_instances, sample_prev)

def evaluate(model, protocol, error_metric, n_jobs=None):
    """Parallel variant of quapy.evaluation.evaluate"""
    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)
    true_prevs, estim_prevs = prediction(model, protocol, n_jobs=n_jobs)
    return error_metric(true_prevs, estim_prevs)

def main():
    print("Loading a subsample of the LeQua data")
    trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")
    trn_data = trn_data.split_stratified(5000, random_state=123)[0] # subsample
    val_gen.true_prevs.df = val_gen.true_prevs.df[:8] # use a subset
    tst_gen.true_prevs.df = tst_gen.true_prevs.df[:8]

    print("Fitting a qunfold method")
    m = QuaPyWrapper(PACC(CVClassifier(LogisticRegression(), 5), seed=123))
    m.fit(trn_data)

    qp_times = []
    my_times = []
    for i in range(4):
        if i > 0:
            print(f"Evaluating the quantifier (trial {i}/3)")
        else:
            print(f"Evaluating the quantifier (this warm-up trial is ignored)")

        start = time.time()
        my_error = evaluate(m, protocol=tst_gen, error_metric="mae", n_jobs=None)
        if i > 0:
            my_times.append(time.time() - start)

        start = time.time()
        qp_error = qp.evaluation.evaluate(m, protocol=tst_gen, error_metric="mae")
        if i > 0:
            qp_times.append(time.time() - start)

        if i > 0:
            print(
                f"- my evaluate:            MAE = {my_error} ({my_times[-1]} s)",
                f"- qp.evaluation.evaluate: MAE = {qp_error} ({qp_times[-1]} s)",
                sep = "\n",
            )
    print(
        "Average times:",
        f"- my evaluate:            {np.mean(my_times)} s",
        f"- qp.evaluation.evaluate: {np.mean(qp_times)} s",
        f"- ratio: {np.mean(my_times) / np.mean(qp_times)}",
        sep = "\n",
    )

if __name__ == '__main__':
    main()
