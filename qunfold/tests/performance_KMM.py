import numpy as np
import time
# from filprofiler.api import profile
from qunfold import KernelTransformer, EnergyKernelTransformer
from qunfold.tests import RNG, make_problem, generate_data

# kernel function for the OldEnergyKernelTransformer
def _energyKernel(X, Y):
    nx = X.shape[0]
    ny = Y.shape[0]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    Y = Y.reshape((1, Y.shape[0], Y.shape[1]))
    norm_x = np.sqrt((X**2).sum(-1)).sum(0) / nx
    norm_y = np.sqrt((Y**2).sum(-1)).sum(1) / ny
    Dlk = np.sqrt(((X - Y)**2).sum(-1))
    return np.squeeze(norm_x + norm_y) - Dlk.sum(0).sum(0) / (nx * ny)

class OldEnergyKernelTransformer(KernelTransformer):
  def __init__(self):
    KernelTransformer.__init__(self, _energyKernel)

def main():
  transformers = [
    EnergyKernelTransformer(),
    OldEnergyKernelTransformer(),
  ]
  results = {} # mapping from class names to prediction times
  n_unequal = 0
  for _ in range(10):
    q, M, p_trn = make_problem()
    X_trn, y_trn = generate_data(M, p_trn)
    p_tst = RNG.permutation(p_trn)
    X_tst, y_tst = generate_data(M, p_tst)
    M_ref = None
    q_ref = None
    fX_ref = None
    for i, t in enumerate(transformers):
      name = t.__class__.__name__
      M = t.fit_transform(X_trn, y_trn)
      q = t.transform(X_tst)
      if M_ref is None:
        M_ref = M
        q_ref = q
      else:
        if not np.allclose(M, M_ref):
          print("M:", name)
          print(M_ref)
          print(M)
          print()
          n_unequal += 1
        if not np.allclose(q, q_ref):
          print("q:", name)
          print(q_ref)
          print(q)
          print()
          n_unequal += 1
      start = time.time()
      def predict_often():
        for _ in range(20): # profiling: predict 20 times
          q = t.transform(X_tst)
      predict_often()
      # profile(predict_often, f"/tmp/fil-{name}") # open /tmp/fil-{name}/index.html
      times = results.get(name, [])
      times.append(time.time() - start)
      results[name] = times
  for name, times in results.items():
    print(f"{name}: {np.mean(times) / 20}")
  print(f"{n_unequal} unequal results between methods")

if __name__ == '__main__':
  main()
