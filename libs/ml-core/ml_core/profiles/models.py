"""Data models for engineer profiles."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class EngineerProfile:
  """Represents an engineer's skill profile."""

  engineer_id: str
  embedding: np.ndarray  # 384-dimensional vector
  keywords: dict[str, int] = field(default_factory=dict)  # keyword -> count
  tickets_completed: int = 0
  last_updated: datetime | None = None
  embedding_model: str = "all-MiniLM-L6-v2"

  def to_dict(self) -> dict:
    """Convert profile to dictionary for storage."""
    return {
      "engineer_id": self.engineer_id,
      "embedding": self.embedding.tolist(),
      "keywords": self.keywords,
      "tickets_completed": self.tickets_completed,
      "last_updated": (self.last_updated.isoformat() if self.last_updated else None),
      "embedding_model": self.embedding_model,
    }

  @classmethod
  def from_dict(cls, data: dict) -> "EngineerProfile":
    """Create profile from dictionary."""
    return cls(
      engineer_id=data["engineer_id"],
      embedding=np.array(data["embedding"]),
      keywords=data.get("keywords", {}),
      tickets_completed=data.get("tickets_completed", 0),
      last_updated=(
        datetime.fromisoformat(data["last_updated"])
        if data.get("last_updated")
        else None
      ),
      embedding_model=data.get("embedding_model", "all-MiniLM-L6-v2"),
    )
