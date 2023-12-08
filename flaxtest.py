import flax # neural networks with JAX
import jax
import jax.numpy as jnp
import numpy as np
import optax # common loss functions and optimizers
import quapy as qp
from clu import metrics # utilities for handling metrics in training loops
from flax import linen as nn
from flax.training import train_state
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

class SimpleModule(nn.Module):
  """A simple model."""
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=512)(x)
    x = nn.relu(x)
    x = nn.Dense(features=28)(x) # the LeQua T1B data has 28 classes
    return x

# a storage for all state involved in training, including metrics
@flax.struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output("loss") # average loss
class TrainingState(train_state.TrainState):
  metrics: Metrics

def create_training_state(module, rng, learning_rate, momentum):
  """Create an initial `TrainingState`."""
  return TrainingState.create(
    apply_fn = module.apply,
    params = module.init( # initialize parameters
      rng,
      jnp.ones((1, 300)) # a template batch with one sample, 300 dimensions
    )["params"],
    tx = optax.sgd(learning_rate, momentum),
    metrics = Metrics.empty()
  )

@jax.jit
def training_step(state, batch):
  """Perform a single parameter update."""
  def loss_fn(params):
    logits = state.apply_fn({ "params": params }, batch["X"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits = logits,
      labels = batch["y"]
    ).mean()
    return loss
  return state.apply_gradients(grads=jax.grad(loss_fn)(state.params)) # update the state

@jax.jit
def compute_metrics(*, state, batch): # could receive either a training or a testing state
  logits = state.apply_fn({ "params": state.params }, batch["X"])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits = logits,
    labels = batch["y"]
  ).mean()
  metric_updates = state.metrics.single_from_model_output(
    logits = logits,
    labels = batch["y"],
    loss = loss
  )
  metrics = state.metrics.merge(metric_updates)
  return state.replace(metrics=metrics) # update the state with metrics

def main(
    n_epochs = 20,
    batch_size = 32,
  ):
  trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")
  X_trn, X_tst, y_trn, y_tst = train_test_split(*trn_data.Xy, test_size=.33, random_state=25)

  # baseline performance ~ 0.78
  print("Computing the baseline accuracy of GridSearchCV(LogisticRegression())")
  baseline_accuracy = GridSearchCV(
    LogisticRegression(),
    { "C":[1e-3, 1e-2, 1e-1, 1e0, 1e1] }
  ).fit(X_trn, y_trn).score(X_tst, y_tst)
  print(f"Baseline accuracy: {baseline_accuracy:.5f}")

  # instantiate the model
  module = SimpleModule()
  print(module.tabulate( # inspect the structure of the model
    jax.random.key(0),
    jnp.ones((1, 300)),
    compute_flops = True,
    compute_vjp_flops = True
  ))
  training_state = create_training_state(
    module,
    jax.random.key(0),
    learning_rate = .01,
    momentum = .9
  )

  # take out the training
  X_trn = jnp.array(X_trn, dtype=jnp.float32)
  y_trn = jnp.array(y_trn)
  shuffle_rng = np.random.default_rng(25)
  metrics_history = {
    "trn_loss": [],
    "trn_accuracy": [],
    "tst_loss": [],
    "tst_accuracy": []
  }
  for epoch in range(n_epochs):

    # take out the training steps of this epoch
    i_epoch = shuffle_rng.permutation(len(X_trn))
    for step in range(len(X_trn) // batch_size):
      i_step = i_epoch[(step * batch_size):((step+1) * batch_size)]
      batch = { "X": X_trn[i_step], "y": y_trn[i_step] }

      # update parameters and metrics
      training_state = training_step(training_state, batch)
      training_state = compute_metrics(state=training_state, batch=batch)

    # compute average training metrics for this epoch and reset the metric state
    for metric, value in training_state.metrics.compute().items():
      metrics_history[f"trn_{metric}"].append(value)
    training_state = training_state.replace(metrics=training_state.metrics.empty()) # reset

    # compute average metrics on the test set
    test_state = training_state
    for step in range(int(np.ceil(len(X_tst) / batch_size))):
      test_state = compute_metrics(state=test_state, batch={
        "X": X_tst[(step * batch_size):np.min(((step+1) * batch_size, len(X_tst)))],
        "y": y_tst[(step * batch_size):np.min(((step+1) * batch_size, len(X_tst)))]
      })
    for metric, value in test_state.metrics.compute().items():
      metrics_history[f"tst_{metric}"].append(value)

    # log information
    print(
      f"[{epoch+1:2d}/{n_epochs}]",
      f"trn_loss={metrics_history['trn_loss'][-1]:.5f},",
      f"trn_acc={metrics_history['trn_accuracy'][-1]:.5f},",
      f"tst_loss={metrics_history['tst_loss'][-1]:.5f},",
      f"tst_acc={metrics_history['tst_accuracy'][-1]:.5f}",
    )

if __name__ == "__main__":
  main()
