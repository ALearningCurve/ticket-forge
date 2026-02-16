"""Temporal feature computation for tickets."""

import pandas as pd


def compute_business_completion_hours(
  created_at: str | None, closed_at: str | None
) -> float | None:
  """Compute completion time in business hours.

  Args:
      created_at: ISO datetime string when ticket was created
      closed_at: ISO datetime string when ticket was closed

  Returns:
      Business hours or None if closed_at is missing
  """
  if closed_at is None or pd.isna(closed_at):
    return None

  if created_at is None or pd.isna(created_at):
    return None

  try:
    start = pd.to_datetime(created_at)
    end = pd.to_datetime(closed_at)
    total_hours = (end - start).total_seconds() / 3600
    business_hours = total_hours * (5 / 7)
    return round(business_hours, 2)
  except (ValueError, TypeError):
    return None
