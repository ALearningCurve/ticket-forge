import pandas as pd
import numpy as np


def compute_business_completion_hours(created_at, closed_at):
    if closed_at is None or pd.isna(closed_at):
        return None

    start = pd.to_datetime(created_at)
    end = pd.to_datetime(closed_at)

    if end <= start:
        return 0.0

    business_days = np.busday_count(
        start.date(),
        end.date()
    )

    # Add partial day hours
    total_hours = business_days * 8
    remainder = (end - start).total_seconds() / 3600
    return max(total_hours, remainder)
