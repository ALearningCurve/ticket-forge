import pandas as pd
import json
import gzip
from datetime import datetime

INPUT_CSV = "tickets_raw.csv"
OUTPUT_JSON = "tickets_final.json.gz"

print("📄 Loading CSV...")
df = pd.read_csv(INPUT_CSV)

print("🔄 Converting to JSON structure...")

json_output = {
    "board": {
        "source": "github_issues",
        "generated_from": INPUT_CSV,
        "generated_at": datetime.utcnow().isoformat(),
        "total_tickets": len(df)
    },
    "tickets": df.to_dict(orient="records")
}

print("📦 Writing compressed JSON...")

with gzip.open(OUTPUT_JSON, "wt", encoding="utf-8") as f:
    json.dump(json_output, f, indent=2)

print(f"✅ Created {OUTPUT_JSON}")
