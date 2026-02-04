"""Tests for Dataset class."""

from ml_core import Dataset


class TestDatasetSubsetLoading:
  """Tests for Dataset subset loading functionality."""

  def test_dataset_load_x_without_subset(self) -> None:
    """Test loading full x dataset without subset."""
    dataset = Dataset(split="train")
    x = dataset.load_x()
    assert x.shape[0] > 0
    assert x.shape[1] == 1
    assert len(x.shape) == 2

  def test_dataset_load_y_without_subset(self) -> None:
    """Test loading full y dataset without subset."""
    dataset = Dataset(split="train")
    y = dataset.load_y()
    assert y.shape[0] > 0
    assert len(y.shape) == 1

  def test_dataset_load_x_with_subset(self) -> None:
    """Test loading x dataset with subset size."""
    dataset = Dataset(split="train", subset_size=20)
    x = dataset.load_x()
    assert x.shape[0] == 20

  def test_dataset_load_y_with_subset(self) -> None:
    """Test loading y dataset with subset size."""
    dataset = Dataset(split="train", subset_size=20)
    y = dataset.load_y()
    assert y.shape[0] == 20

  def test_dataset_subset_consistency(self) -> None:
    """Test that x and y subset sizes are consistent."""
    dataset = Dataset(split="train", subset_size=15)
    x = dataset.load_x()
    y = dataset.load_y()
    assert x.shape[0] == y.shape[0] == 15

  def test_as_sklearn_cv_split_without_subset(self) -> None:
    """Test creating sklearn CV split without subset."""
    x, y, cv_split = Dataset.as_sklearn_cv_split()
    assert x.shape[0] == 200  # train + validation (100 + 100)
    assert y.shape[0] == 200
    assert cv_split is not None

  def test_as_sklearn_cv_split_with_subset(self) -> None:
    """Test creating sklearn CV split with subset."""
    x, y, cv_split = Dataset.as_sklearn_cv_split(subset_size=10)
    assert x.shape[0] == 20  # train + validation (10 + 10)
    assert y.shape[0] == 20
    assert cv_split is not None
