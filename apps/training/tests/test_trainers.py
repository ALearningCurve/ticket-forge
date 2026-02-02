"""Tests for training models using different trainers."""

import pytest
from sklearn.model_selection import RandomizedSearchCV
from training.trainers.train_forest import fit_grid as fit_grid_forest
from training.trainers.train_linear import fit_grid as fit_grid_linear
from training.trainers.train_svm import fit_grid_approx as fit_grid_svm
from training.trainers.train_xgboost import fit_grid as fit_grid_xgboost
from training.trainers.utils.harness import Dataset


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


class TestForestTrainer:
  """Tests for Random Forest trainer."""

  @pytest.mark.filterwarnings("ignore")
  def test_forest_trainer_fits_successfully(self) -> None:
    """Test that forest trainer can fit on small dataset."""
    x_combined, y_combined, cv_split = Dataset.as_sklearn_cv_split(subset_size=20)

    grid = fit_grid_forest(x_combined, y_combined, cv_split)

    assert isinstance(grid, RandomizedSearchCV)
    assert grid.best_estimator_ is not None
    assert hasattr(grid, "cv_results_")

  @pytest.mark.filterwarnings("ignore")
  def test_forest_trainer_predictions(self) -> None:
    """Test that forest trainer can make predictions."""
    x_combined, y_combined, cv_split = Dataset.as_sklearn_cv_split(subset_size=20)

    grid = fit_grid_forest(x_combined, y_combined, cv_split)
    test_dataset = Dataset(split="test", subset_size=20)
    x_test = test_dataset.load_x()
    predictions = grid.predict(x_test)

    assert predictions.shape[0] == 20


class TestLinearTrainer:
  """Tests for Linear trainer."""

  @pytest.mark.filterwarnings("ignore")
  def test_linear_trainer_fits_successfully(self) -> None:
    """Test that linear trainer can fit on small dataset."""
    x_combined, y_combined, cv_split = Dataset.as_sklearn_cv_split(subset_size=20)

    grid = fit_grid_linear(x_combined, y_combined, cv_split)

    assert isinstance(grid, RandomizedSearchCV)
    assert grid.best_estimator_ is not None
    assert hasattr(grid, "cv_results_")

  @pytest.mark.filterwarnings("ignore")
  def test_linear_trainer_predictions(self) -> None:
    """Test that linear trainer can make predictions."""
    x_combined, y_combined, cv_split = Dataset.as_sklearn_cv_split(subset_size=20)

    grid = fit_grid_linear(x_combined, y_combined, cv_split)
    test_dataset = Dataset(split="test", subset_size=20)
    x_test = test_dataset.load_x()
    predictions = grid.predict(x_test)

    assert predictions.shape[0] == 20


class TestSVMTrainer:
  """Tests for SVM trainer with kernel approximation."""

  @pytest.mark.filterwarnings("ignore")
  def test_svm_trainer_fits_successfully(self) -> None:
    """Test that SVM trainer can fit on small dataset."""
    x_combined, y_combined, cv_split = Dataset.as_sklearn_cv_split(subset_size=20)

    grid = fit_grid_svm(x_combined, y_combined, cv_split)

    assert isinstance(grid, RandomizedSearchCV)
    assert grid.best_estimator_ is not None
    assert hasattr(grid, "cv_results_")

  @pytest.mark.filterwarnings("ignore")
  def test_svm_trainer_predictions(self) -> None:
    """Test that SVM trainer can make predictions."""
    x_combined, y_combined, cv_split = Dataset.as_sklearn_cv_split(subset_size=20)

    grid = fit_grid_svm(x_combined, y_combined, cv_split)
    test_dataset = Dataset(split="test", subset_size=20)
    x_test = test_dataset.load_x()
    predictions = grid.predict(x_test)

    assert predictions.shape[0] == 20


class TestXGBoostTrainer:
  """Tests for XGBoost trainer."""

  @pytest.mark.filterwarnings("ignore")
  def test_xgboost_trainer_fits_successfully(self) -> None:
    """Test that XGBoost trainer can fit on small dataset."""
    x_combined, y_combined, cv_split = Dataset.as_sklearn_cv_split(subset_size=20)

    grid = fit_grid_xgboost(x_combined, y_combined, cv_split)

    assert isinstance(grid, RandomizedSearchCV)
    assert grid.best_estimator_ is not None
    assert hasattr(grid, "cv_results_")

  @pytest.mark.filterwarnings("ignore")
  def test_xgboost_trainer_predictions(self) -> None:
    """Test that XGBoost trainer can make predictions."""
    x_combined, y_combined, cv_split = Dataset.as_sklearn_cv_split(subset_size=20)

    grid = fit_grid_xgboost(x_combined, y_combined, cv_split)
    test_dataset = Dataset(split="test", subset_size=20)
    x_test = test_dataset.load_x()
    predictions = grid.predict(x_test)

    assert predictions.shape[0] == 20
