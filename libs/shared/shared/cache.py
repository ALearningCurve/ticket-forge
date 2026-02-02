import json
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import joblib

P = ParamSpec("P")
R = TypeVar("R")


class Saver(ABC):
  """Interface representing different backends to cache with."""

  @abstractmethod
  def dump(self, obj: R, path: Path) -> None:
    """Dumps the given object to the specified path in JSON format.

    Args:
        obj: The object to dump.
        path: The path where the object will be saved.
    """
    pass

  @abstractmethod
  def load(self, path: Path) -> R:
    """Loads from given path.

    Args:
        path: the path to load from

    Returns:
        the loaded item
    """
    pass


class JoblibSaver(Saver):
  """A saver that uses joblib to serialize and deserialize objects."""

  def dump(self, obj: R, path: Path) -> None:
    """Dumps obj to given path.

    Args:
        obj: the object to dump
        path: the path to dump to
    """
    joblib.dump(obj, path)

  def load(self, path: Path) -> R:
    """Loads from given path.

    Args:
        path: the path to load from

    Returns:
        the loaded item
    """
    return joblib.load(path)


class JsonSaver(Saver):
  """A saver that users json to serialize and deserialize objects."""

  def dump(self, obj: R, path: Path) -> None:
    """Dumps obj to given path.

    Args:
        obj: the object to dump
        path: the path to dump to
    """
    with open(path, "w") as f:
      json.dump(obj, f)

  def load(self, path: Path) -> R:
    """Loads from given path.

    Args:
        path: the path to load from

    Returns:
        the loaded item
    """
    with open(path, "r") as f:
      return json.load(f)


DEFAULT_SAVER = JoblibSaver()


def fs_cache(
  cache_path: Path, saver: Saver = DEFAULT_SAVER
) -> Callable[[Callable[P, R]], Callable[P, R]]:
  """Decorator to cache function results using joblib."""

  def decorator(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
      if cache_path.exists():
        print(f"Loading cached result from: {cache_path}")
        return saver.load(cache_path)  # type: ignore

      result = func(*args, **kwargs)

      cache_path.parent.mkdir(parents=True, exist_ok=True)
      print(f"Saving result to: {cache_path}")
      saver.dump(result, cache_path)  # pyright: ignore[reportUnknownMemberType]

      return result

    return wrapper

  return decorator
