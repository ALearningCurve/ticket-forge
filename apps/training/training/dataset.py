"""Dataset utilities for machine learning pipelines."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel
from shared.configuration import RANDOM_SEED, TRAIN_USE_DUMMY_DATA, Paths, Splits_t
from sklearn.datasets import make_regression
from sklearn.model_selection import PredefinedSplit

X_t = npt.NDArray[Any]
Y_t = npt.NDArray[np.floating]

# Split ratios for train / validation / test
_SPLIT_RATIOS: dict[str, float] = {"train": 0.7, "validation": 0.15, "test": 0.15}


def _find_latest_pipeline_output() -> Path:
  """Returns the path to the most recent timestamped pipeline output directory
  that contains a tickets_transformed_improved.jsonl file.

  Looks for directories matching 'github_issues-*' under data_root, sorted
  lexicographically (ISO timestamps sort correctly this way), skipping any
  incomplete directories that lack the required data file. Falls back to
  the legacy 'github_issues' directory if no valid timestamped run is found.

  Returns:
      Path to the latest valid pipeline output directory.

  Raises:
      FileNotFoundError: If no valid pipeline output directory can be located.
  """
  data_root = Paths.data_root
  required_file = "tickets_transformed_improved.jsonl"
  timestamped = sorted(data_root.glob("github_issues-*"), reverse=True)
  for candidate in timestamped:
    if (candidate / required_file).exists():
      return candidate
  legacy = data_root / "github_issues"
  if (legacy / required_file).exists():
    return legacy
  msg = (
    f"No valid pipeline output found under {data_root}. "
    "Run the ticket_etl DAG or scraper first."
  )
  raise FileNotFoundError(msg)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  """Loads a .jsonl file into a list of dicts.

  Args:
      path: Path to the .jsonl file.

  Returns:
      List of parsed JSON objects.

  Raises:
      FileNotFoundError: If the file does not exist.
  """
  if not path.exists():
    msg = f"Expected data file not found: {path}"
    raise FileNotFoundError(msg)
  records = []
  with open(path) as f:
    for line in f:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  return records


def _split_indices(
  n: int, split: Splits_t, seed: int = RANDOM_SEED
) -> npt.NDArray[np.intp]:
  """Returns row indices for the requested split.

  Shuffles deterministically using seed, then slices according to
  _SPLIT_RATIOS so every split sees a different, non-overlapping subset.

  Args:
      n:     Total number of records.
      split: One of 'train', 'validation', or 'test'.
      seed:  Random seed for reproducibility.

  Returns:
      Array of integer indices for the requested split.
  """
  rng = np.random.default_rng(seed=seed)
  idx = rng.permutation(n)

  train_end = int(n * _SPLIT_RATIOS["train"])
  val_end = train_end + int(n * _SPLIT_RATIOS["validation"])

  if split == "train":
    return idx[:train_end]
  if split == "validation":
    return idx[train_end:val_end]
  # test
  return idx[val_end:]


class Dataset(BaseModel):
  """Represents training dataset for ticket time prediction.

  Loads real pipeline output from the latest timestamped run directory
  under data_root when TRAIN_USE_DUMMY_DATA is False. Falls back to
  synthetically generated data when TRAIN_USE_DUMMY_DATA is True.
  """

  split: Splits_t
  subset_size: int | None = None

  # ------------------------------------------------------------------ #
  # Internal helpers                                                     #
  # ------------------------------------------------------------------ #

  def _load_records(self) -> list[dict[str, Any]]:
    """Loads and splits the transformed ticket records for this split.

    Returns:
        List of ticket dicts belonging to this split.
    """
    pipeline_dir = _find_latest_pipeline_output()
    jsonl_path = pipeline_dir / "tickets_transformed_improved.jsonl"
    all_records = _load_jsonl(jsonl_path)

    indices = _split_indices(len(all_records), self.split)
    records = [all_records[i] for i in indices]

    if self.subset_size is not None:
      records = records[: self.subset_size]
    return records

  # ------------------------------------------------------------------ #
  # Public loaders                                                       #
  # ------------------------------------------------------------------ #

  def load_x(self) -> X_t:
    """Loads the feature matrix X.

    Each row is the 384-dimensional embedding vector for one ticket,
    as produced by the all-MiniLM-L6-v2 model in the transform stage.

    Returns:
        Float32 array of shape (n_samples, 384).
    """
    if TRAIN_USE_DUMMY_DATA:
      dataset = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
      x = dataset[0]
      if self.subset_size is not None:
        return x[: self.subset_size]  # type: ignore[return-value]
      return x  # type: ignore[return-value]

    records = self._load_records()
    embeddings = [r["embedding"] for r in records]
    return np.array(embeddings, dtype=np.float32)

  def load_y(self) -> Y_t:
    """Loads the target vector y.

    Target is completion_hours_business — the number of business hours
    between ticket assignment and closure, as computed by the transform
    stage. Tickets with missing values are replaced with the column mean
    so downstream models always receive a complete target vector.

    Returns:
        Float64 array of shape (n_samples,).
    """
    if TRAIN_USE_DUMMY_DATA:
      dataset = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
      y = dataset[1]
      if self.subset_size is not None:
        return y[: self.subset_size]  # type: ignore[return-value]
      return y  # type: ignore[return-value]

    records = self._load_records()
    raw = [r.get("completion_hours_business") for r in records]
    y = np.array(raw, dtype=np.float64)

    # Replace missing values with the column mean so the array is complete
    missing_mask = np.isnan(y)
    if missing_mask.any():
      y[missing_mask] = np.nanmean(y)

    return y

  def load_metadata(self) -> pd.DataFrame:
    """Loads metadata for bias analysis (repo, seniority, labels, completion time).

    Returns:
        DataFrame with columns: repo, seniority, labels,
        completion_hours_business.
    """
    if TRAIN_USE_DUMMY_DATA:
      n_samples = self.subset_size if self.subset_size is not None else 100
      rng = np.random.default_rng(seed=42)
      return pd.DataFrame(
        {
          "repo": rng.choice(["terraform", "ansible", "prometheus"], size=n_samples),
          "seniority": rng.choice(["junior", "mid", "senior"], size=n_samples),
          "labels": rng.choice(
            ["bug", "enhancement", "feature", "bug,critical"],
            size=n_samples,
          ),
          "completion_hours_business": rng.uniform(1, 100, size=n_samples),
        }
      )

    records = self._load_records()
    return pd.DataFrame(
      {
        "repo": [r.get("repo", "") for r in records],
        "seniority": [r.get("seniority", "mid") for r in records],
        "labels": [r.get("labels", "") for r in records],
        "completion_hours_business": [
          r.get("completion_hours_business") for r in records
        ],
      }
    )

  def load_sample_weights(self) -> Y_t:
    """Loads per-sample weights for bias-aware training.

    Attempts to load weights from sample_weights.json in the latest
    pipeline output directory. Falls back to inverse-frequency weights
    derived from the metadata repo column if the file is missing or
    malformed.

    Returns:
        Float64 array of per-sample weights, same length as load_y().
    """
    group_col = "repo"
    meta = self.load_metadata()

    # Try loading from the latest pipeline output directory first,
    # then fall back to the legacy fixed path.
    candidates: list[Path] = []
    try:
      candidates.append(_find_latest_pipeline_output() / "sample_weights.json")
    except FileNotFoundError:
      pass
    candidates.append(Paths.data_root / "github_issues" / "sample_weights.json")

    for weights_path in candidates:
      try:
        with open(weights_path) as f:
          data: dict[str, Any] = json.load(f)
        saved_col: str = data.get("group_col", group_col)
        weights_by_group: dict[str, float] = data.get("weights_by_group", {})
        if saved_col in meta.columns and weights_by_group:
          w = meta[saved_col].map(weights_by_group).fillna(1.0)  # type: ignore[arg-type]
          return w.to_numpy(dtype=np.float64)  # type: ignore[return-value]
      except (FileNotFoundError, json.JSONDecodeError):
        continue

    # Fallback: inverse-frequency weights from metadata repo column
    group_counts = meta[group_col].value_counts()
    total = len(meta)
    n_groups = len(group_counts)
    w = meta[group_col].map(lambda g: total / (n_groups * group_counts[g]))
    return w.to_numpy(dtype=np.float64)  # type: ignore[return-value]

  # ------------------------------------------------------------------ #
  # Sklearn CV helpers (unchanged API)                                   #
  # ------------------------------------------------------------------ #

  @staticmethod
  def as_sklearn_cv_split(
    subset_size: int | None = None,
  ) -> tuple[X_t, Y_t, PredefinedSplit]:
    """Creates a predefined sklearn cross-validation split with fixed
    training and validation partitions.

    Returns:
        Tuple of (x_combined, y_combined, cv_split).
    """
    train = Dataset(split="train", subset_size=subset_size)
    validation = Dataset(split="validation", subset_size=subset_size)

    x_train = train.load_x()
    y_train = train.load_y()
    x_val = validation.load_x()
    y_val = validation.load_y()

    x_combined = np.vstack([x_train, x_val])
    y_combined = np.hstack([y_train, y_val])

    test_fold = np.concatenate(
      [np.full(x_train.shape[0], -1), np.full(x_val.shape[0], 0)]
    )
    cv_split = PredefinedSplit(test_fold)

    return x_combined, y_combined, cv_split

  @staticmethod
  def as_sklearn_cv_split_with_weights(
    subset_size: int | None = None,
  ) -> tuple[X_t, Y_t, PredefinedSplit, Y_t]:
    """Like as_sklearn_cv_split but also returns per-sample weights.

    Returns:
        Tuple of (x_combined, y_combined, cv_split, weights_combined).
    """
    train = Dataset(split="train", subset_size=subset_size)
    validation = Dataset(split="validation", subset_size=subset_size)

    x_train, y_train = train.load_x(), train.load_y()
    w_train = train.load_sample_weights()
    x_val, y_val = validation.load_x(), validation.load_y()
    w_val = validation.load_sample_weights()

    x_combined = np.vstack([x_train, x_val])
    y_combined = np.hstack([y_train, y_val])
    w_combined = np.hstack([w_train, w_val])

    test_fold = np.concatenate(
      [np.full(x_train.shape[0], -1), np.full(x_val.shape[0], 0)]
    )
    cv_split = PredefinedSplit(test_fold)

    return x_combined, y_combined, cv_split, w_combined
