# csv_to_json.py

import os
import pandas as pd
import json
import gzip
from datetime import datetime

DATA_DIR = "data/github_issues"
os.makedirs(DATA_DIR, exist_ok=True)

INPUT_CSV = os.path.join(DATA_DIR, "tickets_raw.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "tickets_final.json.gz")

print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)

json_output = {
    "board": {
        "source": "github_issues",
        "issue_state": "closed",
        "generated_at": datetime.utcnow().isoformat(),
        "total_tickets": len(df)
    },
    "tickets": df.to_dict(orient="records")
}

print("Writing compressed JSON...")
with gzip.open(OUTPUT_JSON, "wt", encoding="utf-8") as f:
    json.dump(json_output, f)

print(f"Created {OUTPUT_JSON}")
