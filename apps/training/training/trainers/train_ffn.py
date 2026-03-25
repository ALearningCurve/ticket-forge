"""Feedforward neural network trainer with configurable architecture.

Uses PyTorch for model definition and training, wrapped in a sklearn-compatible
interface for seamless integration with RandomizedSearchCV and the training harness.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import mlflow
import numpy as np
import torch
from shared.configuration import RANDOM_SEED
from shared.logging import get_logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from training.trainers.base import BaseTrainer
from training.trainers.utils.harness import X_t, Y_t, load_fit_debug, load_fit_dump

logger = get_logger(__name__)

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class FFNNet(nn.Module):
  """Configurable feedforward neural network for regression.

  Attributes:
      input_size: Number of input features.
      hidden_sizes: Tuple of hidden layer sizes.
      dropout_rate: Dropout probability (0.0 to 1.0).
      output_size: Number of output dimensions (1 for regression).
  """

  def __init__(
    self,
    input_size: int,
    hidden_sizes: tuple[int, ...],
    dropout_rate: float = 0.2,
    use_batch_norm: bool = True,
    output_size: int = 1,
  ) -> None:
    """Initialize FFN with specified architecture.

    Args:
        input_size: Number of input features.
        hidden_sizes: Tuple of hidden layer sizes (e.g., (128, 64, 32)).
        dropout_rate: Dropout probability between layers.
        use_batch_norm: Whether to add BatchNorm1d after each hidden linear layer.
        output_size: Output dimension (1 for regression).
    """
    super().__init__()
    self.input_size = input_size
    self.hidden_sizes = hidden_sizes
    self.dropout_rate = dropout_rate
    self.use_batch_norm = use_batch_norm
    self.output_size = output_size

    layers: list[nn.Module] = []
    prev_size = input_size

    # Build hidden layers
    for hidden_size in hidden_sizes:
      layers.append(nn.Linear(prev_size, hidden_size))
      if use_batch_norm:
        layers.append(nn.BatchNorm1d(hidden_size))
      layers.append(nn.ReLU())
      if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
      prev_size = hidden_size

    # Output layer
    layers.append(nn.Linear(prev_size, output_size))

    self.net = nn.Sequential(*layers)
    logger.info(f"created {self}")

  def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """Forward pass through the network.

    Args:
        x: Input tensor of shape (batch_size, input_size).

    Returns:
        Output tensor of shape (batch_size, output_size).
    """
    return self.net(x)


class SklearnFFNRegressor(BaseEstimator, RegressorMixin):
  """Sklearn-compatible wrapper around PyTorch FFN for regression.

  Attributes:
      hidden_sizes: Tuple of hidden layer sizes.
      dropout_rate: Dropout probability.
      learning_rate: Adam optimizer learning rate.
      batch_size: Training batch size.
      epochs: Number of training epochs.
      device: 'cpu' or 'cuda'.
      verbose: Whether to print training loss.
  """

  def __init__(  # noqa: PLR0913
    self,
    hidden_sizes: tuple[int, ...] | None = None,
    num_hidden_layers: int = 2,
    hidden_size_1: int = 8,
    hidden_size_2: int = 8,
    hidden_size_3: int = 8,
    hidden_size_4: int = 8,
    dropout_rate: float = 0.1,
    use_batch_norm: bool = True,
    weight_decay: float = 1e-2,
    learning_rate: float = 0.0001,
    batch_size: int = 32,
    epochs: int = 300,
    device: str = "cpu",
    verbose: bool = True,
    validation_split: float = 0.1,
    eval_every_epochs: int = 1,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 1e-4,
  ) -> None:
    """Initialize sklearn FFN regressor.

    Arguments here are intentionally numerous because RandomizedSearchCV needs
    independent hyperparameters to vary over. This is a necessary design choice
    for sklearn compatibility.

    Args:
      hidden_sizes: Explicit tuple of hidden sizes. If None, built from
        num_hidden_layers and hidden_size_1..hidden_size_4.
      num_hidden_layers: Number of hidden layers to use when hidden_sizes is None.
      hidden_size_1: Width of first hidden layer.
      hidden_size_2: Width of second hidden layer.
      hidden_size_3: Width of third hidden layer.
      hidden_size_4: Width of fourth hidden layer.
      dropout_rate: Dropout probability (0.0-1.0).
      use_batch_norm: Whether to apply BatchNorm after hidden linear layers.
      weight_decay: L2 regularization coefficient passed to Adam.
      learning_rate: Adam learning rate.
      batch_size: Batch size for training.
      epochs: Number of full passes over training data.
      device: 'cpu' or 'cuda'.
      verbose: Print training progress.
      validation_split: Fraction of training data reserved for validation.
      eval_every_epochs: Evaluate validation loss every N epochs.
      early_stopping_patience: Stop after this many non-improving evals.
      early_stopping_min_delta: Minimum validation improvement to reset patience.
    """
    self.hidden_sizes = hidden_sizes
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size_1 = hidden_size_1
    self.hidden_size_2 = hidden_size_2
    self.hidden_size_3 = hidden_size_3
    self.hidden_size_4 = hidden_size_4
    self.dropout_rate = dropout_rate
    self.use_batch_norm = use_batch_norm
    self.weight_decay = weight_decay
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.epochs = epochs
    self.device = device
    self.verbose = verbose
    self.validation_split = validation_split
    self.eval_every_epochs = eval_every_epochs
    self.early_stopping_patience = early_stopping_patience
    self.early_stopping_min_delta = early_stopping_min_delta

    self.model_: FFNNet | None = None
    self.scaler_x_: StandardScaler | None = None
    self.scaler_y_: StandardScaler | None = None

  def fit(  # noqa: PLR0912, PLR0915
    self,
    x: X_t,
    y: Y_t,
    sample_weight: Y_t | None = None,
  ) -> SklearnFFNRegressor:
    """Fit the FFN model on training data.

    Args:
        x: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).
        sample_weight: Optional per-sample weights.

    Returns:
        Self for method chaining.
    """
    # Standardize inputs and outputs for stable training
    self.scaler_x_ = StandardScaler()
    x_scaled = self.scaler_x_.fit_transform(x)

    self.scaler_y_ = StandardScaler()
    y_scaled = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()

    # Create model
    n_features = x_scaled.shape[1]
    resolved_hidden_sizes = self._resolve_hidden_sizes()
    self.model_ = FFNNet(
      input_size=n_features,
      hidden_sizes=resolved_hidden_sizes,
      dropout_rate=self.dropout_rate,
      use_batch_norm=self.use_batch_norm,
      output_size=1,
    ).to(self.device)

    # Training loop
    optimizer = torch.optim.AdamW(
      self.model_.parameters(),
      lr=self.learning_rate,
      weight_decay=self.weight_decay,
    )
    criterion = nn.MSELoss()

    x_tensor_all = torch.from_numpy(x_scaled).float().to(self.device)
    y_tensor_all = torch.from_numpy(y_scaled).float().to(self.device)

    if sample_weight is not None:
      weight_tensor_all = torch.from_numpy(sample_weight).float().to(self.device)
    else:
      weight_tensor_all = torch.ones_like(y_tensor_all)

    n_samples = x_tensor_all.shape[0]
    val_size = int(n_samples * self.validation_split)
    if val_size <= 0 or n_samples - val_size < 2:
      val_size = min(1, max(0, n_samples - 2))

    # Deterministic split for reproducibility.
    split_gen = torch.Generator(device="cpu")
    split_gen.manual_seed(RANDOM_SEED)
    indices = torch.randperm(n_samples, generator=split_gen).to(self.device)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    x_train = x_tensor_all[train_indices]
    y_train = y_tensor_all[train_indices]
    w_train = weight_tensor_all[train_indices]
    x_val = x_tensor_all[val_indices]
    y_val = y_tensor_all[val_indices]

    n_train_samples = x_train.shape[0]
    best_val_loss = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    non_improving_evals = 0

    for epoch in range(self.epochs):
      epoch_loss = 0.0
      n_batches = 0

      # Shuffle and batch
      train_perm = torch.randperm(n_train_samples, device=self.device)
      for i in range(0, n_train_samples, self.batch_size):
        batch_indices = train_perm[i : i + self.batch_size]
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]
        w_batch = w_train[batch_indices]

        optimizer.zero_grad()
        y_pred = self.model_(x_batch).squeeze()
        loss = criterion(y_pred, y_batch)

        # Apply sample weights
        weighted_loss = (loss * w_batch).mean()
        weighted_loss.backward()
        optimizer.step()

        epoch_loss += weighted_loss.item()
        n_batches += 1
      avg_loss = epoch_loss / n_batches
      should_eval = (epoch + 1) % self.eval_every_epochs == 0 or (
        epoch + 1
      ) == self.epochs

      if should_eval and val_size > 0:
        self.model_.eval()
        with torch.no_grad():
          val_pred = self.model_(x_val).squeeze()
          val_loss = criterion(val_pred, y_val).item()

        improved = (best_val_loss - val_loss) > self.early_stopping_min_delta
        if improved:
          best_val_loss = val_loss
          best_model_state = copy.deepcopy(self.model_.state_dict())
          non_improving_evals = 0
        else:
          non_improving_evals += 1

        if self.verbose:
          logger.info(
            "Epoch %d/%d, train_loss=%.6f, val_loss=%.6f, no_improve=%d",
            epoch + 1,
            self.epochs,
            avg_loss,
            val_loss,
            non_improving_evals,
          )

        if non_improving_evals >= self.early_stopping_patience:
          if self.verbose:
            msg = (
              "Early stopping at epoch %d "
              "(validation plateau/worsening). best_val_loss=%.6f"
            )
            logger.info(msg, epoch + 1, best_val_loss)
          break

        self.model_.train()
      elif self.verbose:
        logger.info("Epoch %d/%d, train_loss=%.6f", epoch + 1, self.epochs, avg_loss)

    if best_model_state is not None:
      self.model_.load_state_dict(best_model_state)
      if self.verbose:
        msg = "Restored best validation checkpoint (val_loss=%.6f)"
        logger.info(msg, best_val_loss)

    return self

  def _resolve_hidden_sizes(self) -> tuple[int, ...]:
    """Resolve hidden layer widths from explicit tuple or tunable params."""
    if self.hidden_sizes is not None:
      return self.hidden_sizes

    layer_sizes = (
      self.hidden_size_1,
      self.hidden_size_2,
      self.hidden_size_3,
      self.hidden_size_4,
    )
    n_layers = max(1, min(self.num_hidden_layers, len(layer_sizes)))
    return tuple(int(size) for size in layer_sizes[:n_layers])

  def predict(self, x: X_t) -> np.ndarray:
    """Predict using the trained FFN.

    Args:
        x: Feature matrix of shape (n_samples, n_features).

    Returns:
        Predicted values of shape (n_samples,).
    """
    if self.model_ is None or self.scaler_x_ is None or self.scaler_y_ is None:
      msg = "Model must be fitted before calling predict"
      raise RuntimeError(msg)

    self.model_.eval()
    x_scaled = self.scaler_x_.transform(x)
    x_tensor = torch.from_numpy(x_scaled).float().to(self.device)

    with torch.no_grad():
      y_scaled = self.model_(x_tensor).squeeze().cpu().numpy()

    # Inverse-transform back to original scale
    return self.scaler_y_.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

  def get_params(self, deep: bool = True) -> dict[str, Any]:
    """Get parameters for sklearn compatibility.

    Args:
        deep: If True, return parameters recursively.

    Returns:
        Parameter dict for sklearn cloning.
    """
    return {
      "hidden_sizes": self.hidden_sizes,
      "num_hidden_layers": self.num_hidden_layers,
      "hidden_size_1": self.hidden_size_1,
      "hidden_size_2": self.hidden_size_2,
      "hidden_size_3": self.hidden_size_3,
      "hidden_size_4": self.hidden_size_4,
      "dropout_rate": self.dropout_rate,
      "use_batch_norm": self.use_batch_norm,
      "weight_decay": self.weight_decay,
      "learning_rate": self.learning_rate,
      "batch_size": self.batch_size,
      "epochs": self.epochs,
      "device": self.device,
      "verbose": self.verbose,
      "validation_split": self.validation_split,
      "eval_every_epochs": self.eval_every_epochs,
      "early_stopping_patience": self.early_stopping_patience,
      "early_stopping_min_delta": self.early_stopping_min_delta,
    }

  def set_params(self, **params: dict[str, Any]) -> SklearnFFNRegressor:
    """Set parameters for sklearn compatibility.

    Args:
        params: Parameter dict.

    Returns:
        Self.
    """
    for key, value in params.items():
      setattr(self, key, value)
    return self


class FFNTrainer(BaseTrainer):
  """Trainer for PyTorch-based Feedforward Neural Network."""

  def get_model(self) -> BaseEstimator:
    """Create an unfitted SklearnFFNRegressor with default parameters.

    Returns:
        SklearnFFNRegressor instance with sensible defaults.
    """
    return SklearnFFNRegressor(verbose=True)

  def get_search_params(self) -> list[dict[str, Any]]:
    """Return hyperparameter search space for FFN.

    Searches over 8 architecture × 4 dropout × 4 lr × 3 batch × 3 epoch
    = 1,152 possible combinations. RandomizedSearchCV samples 50 random.

    Returns:
        List containing a single param distribution dict.
    """
    return [
      {
        "hidden_sizes": [
          (64,),
          (128,),
          (256,),
          (128, 64),
          (256, 128),
          (256, 128, 64),
          (512, 256),
          (512, 256, 128),
        ],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4],
        "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "batch_size": [16, 32, 64],
        "epochs": [50, 100, 200],
      }
    ]

  def get_search_config(self) -> dict[str, Any]:
    """Override search config for FFN.

    Uses n_jobs=1 since pytorch neural network training doesn't parallelize well
    across multiple CV splits. Better to do sequential evaluation.

    Returns:
        Config dict with n_iter=50, n_jobs=1.
    """
    config = super().get_search_config()
    config["n_iter"] = 50
    config["n_jobs"] = 1  # Neural network training doesn't parallelize well
    return config


def fit_grid(
  x: X_t,
  y: Y_t,
  cv_split: PredefinedSplit,
  sample_weight: Y_t | None = None,
) -> object:
  """Performs hyperparameter search for FFN regressor using Optuna.

  Args:
      x: Feature matrix.
      y: Target values.
      cv_split: PredefinedSplit for cross-validation.
      sample_weight: Optional per-sample weights for bias-aware training.

  Returns:
      Fitted OptunaSearchCV-like object compatible with harness usage.
  """
  _ensure_mlflow_experiment_context()

  try:
    import optuna
    from optuna.distributions import (
      CategoricalDistribution,
      FloatDistribution,
      IntDistribution,
    )
    from optuna_integration.sklearn import OptunaSearchCV
  except ModuleNotFoundError as err:
    msg = (
      "Optuna dependencies are missing. Run `just` to sync workspace dependencies "
      "after updating apps/training/pyproject.toml."
    )
    raise RuntimeError(msg) from err

  param_distributions = {
    "num_hidden_layers": IntDistribution(1, 4),
    "hidden_size_1": IntDistribution(32, 128, step=32),
    "hidden_size_2": IntDistribution(32, 128, step=32),
    "hidden_size_3": IntDistribution(32, 128, step=32),
    "hidden_size_4": IntDistribution(32, 128, step=32),
    "dropout_rate": FloatDistribution(0.1, 0.4, step=0.1),
    "use_batch_norm": CategoricalDistribution([True, False]),
    "weight_decay": FloatDistribution(1e-8, 1e-2, log=True),
    "learning_rate": FloatDistribution(1e-4, 5e-3, log=True),
    "batch_size": CategoricalDistribution([16, 32, 64]),
    "epochs": IntDistribution(50, 200),
  }

  study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
  )

  search = OptunaSearchCV(
    estimator=SklearnFFNRegressor(device="cpu", verbose=True),
    param_distributions=param_distributions,
    cv=cv_split,
    scoring="neg_mean_squared_error",
    n_trials=10,
    n_jobs=1,
    refit=True,
    error_score="raise",
    study=study,
    callbacks=[_optuna_mlflow_callback],
  )

  # PredefinedSplit in this pipeline has a single validation fold; Optuna's
  # Terminator callback expects >1 fold scores and emits a benign warning.
  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore",
      message=r"Failed to report cross validation scores for TerminatorCallback.*",
      category=UserWarning,
      module=r"optuna_integration\.sklearn\.sklearn",
    )
    fitted = search.fit(x, y, sample_weight=sample_weight)
  _ensure_cv_results_compatibility(fitted)
  return fitted


def _ensure_mlflow_experiment_context() -> None:
  """Validate that a parent MLflow run/experiment is active before search starts."""
  active_run = mlflow.active_run()
  if active_run is None:
    msg = (
      "FFN Optuna search requires an active MLflow run. "
      "Ensure train.py starts MLflow run before invoking trainer."
    )
    raise RuntimeError(msg)

  experiment_id = active_run.info.experiment_id
  experiment = mlflow.get_experiment(experiment_id)
  if experiment is None:
    msg = f"Active MLflow experiment not found for id={experiment_id}."
    raise RuntimeError(msg)

  logger.info(
    "FFN Optuna search using MLflow experiment %s (%s)",
    experiment.name,
    experiment_id,
  )


def _optuna_mlflow_callback(study: object, trial: object) -> None:
  """Log each Optuna trial into nested MLflow runs."""
  _ = study
  if mlflow.active_run() is None:
    return

  trial_number = getattr(trial, "number", -1)
  trial_state = getattr(getattr(trial, "state", None), "name", "unknown")
  trial_params = getattr(trial, "params", {})
  trial_value = getattr(trial, "value", None)

  with mlflow.start_run(run_name=f"optuna_trial_{trial_number}", nested=True):
    mlflow.set_tag("run_level", "optuna-trial")
    mlflow.set_tag("trial_number", trial_number)
    mlflow.set_tag("trial_state", trial_state)
    for param_name, param_value in trial_params.items():
      mlflow.log_param(param_name, param_value)
    if trial_value is not None:
      mlflow.log_metric("trial_objective", float(trial_value))


def _ensure_cv_results_compatibility(search: object) -> None:
  """Attach cv_results_ on Optuna search objects if missing.

  The training harness pretty-printer expects cv_results_ in a GridSearch-like
  structure. OptunaSearchCV may not always expose this in a fully compatible
  shape, so we build a minimal one from study trials when needed.
  """
  if hasattr(search, "cv_results_"):
    return

  study = getattr(search, "study_", None)
  if study is None:
    return

  trials = [trial for trial in study.trials if trial.value is not None]
  if not trials:
    return

  scores = np.array([float(trial.value) for trial in trials], dtype=float)
  params = [trial.params for trial in trials]
  rank_test_score = np.argsort(np.argsort(-scores)) + 1

  search.cv_results_ = {
    "mean_fit_time": np.zeros(len(trials), dtype=float),
    "mean_score_time": np.zeros(len(trials), dtype=float),
    "params": params,
    "mean_test_score": scores,
    "rank_test_score": rank_test_score,
  }


def fit_simple(
  x: X_t,
  y: Y_t,
  sample_weight: Y_t | None = None,
) -> SklearnFFNRegressor:
  """Train FFN model with default parameters (debug mode).

  Args:
    x: Feature matrix.
    y: Target values.
    sample_weight: Optional per-sample weights for bias-aware training.

  Returns:
      Fitted SklearnFFNRegressor model.
  """
  trainer = FFNTrainer()
  return trainer.fit_simple(x, y, sample_weight)


def main(run_id: str, debug: bool = False) -> None:
  """Trains FFN model on ticket resolution time prediction task.

  Args:
      run_id: Training run identifier.
      debug: If True, skip hyperparameter tuning.
  """
  if debug:
    load_fit_debug(fit_simple, run_id, "ffn")
  else:
    load_fit_dump(fit_grid, run_id, "ffn")


if __name__ == "__main__":
  main("TESTING")
