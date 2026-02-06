from pathlib import Path
from typing import Literal

RANDOM_SEED = 42

TRAIN_USE_DUMMY_DATA = True

Splits_t = Literal["train", "test", "validation"]
SPLITS: list[Splits_t] = ["train", "test", "validation"]


class Paths:
  """Relevant paths in this project."""

  repo_root = Path(__file__).parent.parent.parent.parent
  data_root = repo_root / "data"
  models_root = repo_root / "models"
