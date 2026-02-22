"""Engineer profile update using Experience Decay Method."""

from datetime import datetime

import numpy as np

from ml_core.profiles.models import EngineerProfile


class ProfileUpdater:
  """Update engineer profiles using Experience Decay Method."""

  def __init__(self, alpha: float = 0.95) -> None:
    """Initialize the profile updater.

    Args:
        alpha: Decay weight (0.95 = 95% old profile, 5% new ticket)
              Higher alpha = more memory, slower adaptation
              Lower alpha = less memory, faster adaptation
    """
    if not 0 < alpha < 1:
      msg = f"Alpha must be between 0 and 1, got {alpha}"
      raise ValueError(msg)

    self.alpha = alpha

  def update_on_ticket_completion(
    self,
    engineer: EngineerProfile,
    ticket_embedding: np.ndarray,
    ticket_keywords: list[str],
  ) -> EngineerProfile:
    """Update engineer profile using Experience Decay Method.

    Args:
        engineer: Current engineer profile
        ticket_embedding: Embedding of completed ticket (384-dim)
        ticket_keywords: Keywords extracted from completed ticket

    Returns:
        Updated engineer profile

    Raises:
        ValueError: If embeddings have mismatched dimensions
    """
    # Validate embedding dimensions match
    if engineer.embedding.shape != ticket_embedding.shape:
      msg = (
        f"Embedding dimension mismatch: "
        f"engineer={engineer.embedding.shape}, "
        f"ticket={ticket_embedding.shape}"
      )
      raise ValueError(msg)

    # Part 1: Update embedding using Experience Decay
    new_embedding = (
      self.alpha * engineer.embedding + (1 - self.alpha) * ticket_embedding
    )

    # Part 2: Update keywords with frequency counts
    new_keywords = engineer.keywords.copy()
    for keyword in ticket_keywords:
      new_keywords[keyword] = new_keywords.get(keyword, 0) + 1

    # Update metadata
    engineer.embedding = new_embedding
    engineer.keywords = new_keywords
    engineer.tickets_completed += 1
    engineer.last_updated = datetime.now()

    return engineer

  def get_decay_influence(self, tickets_completed: int) -> float:
    """Calculate how much influence the original resume still has.

    Args:
        tickets_completed: Number of tickets engineer has completed

    Returns:
        Percentage influence (0.0 to 1.0) of original resume
    """
    return self.alpha**tickets_completed
