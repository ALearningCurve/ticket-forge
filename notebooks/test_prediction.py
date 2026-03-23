"""
Test model predictions on newly created tickets.
Run from repo root: python notebooks/test_prediction.py
"""
import sys
import joblib
import numpy as np
from pathlib import Path

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps/training"))
sys.path.insert(0, str(REPO_ROOT / "libs/shared"))

from ml_core.embeddings.service import EmbeddingService  # noqa: E402

# ── Load best model ──────────────────────────────────────────────────────────
models_dir = REPO_ROOT / "models"
runs = sorted([r for r in models_dir.iterdir() if r.name[0].isdigit() or r.name.startswith("cap")], reverse=True)
latest_run = runs[0]
print(f"Using run: {latest_run.name}\n")

model_path = latest_run / "random_forest.pkl"
model = joblib.load(model_path)

# ── Embedding service ────────────────────────────────────────────────────────
embedder = EmbeddingService()

# ── Feature builder ──────────────────────────────────────────────────────────
REPOS = ["ansible/ansible", "hashicorp/terraform", "prometheus/prometheus"]

def build_features(ticket: dict) -> np.ndarray:
    """Build feature vector matching load_x() in dataset.py."""
    # Embedding from normalized text
    text = f"{ticket['title']} {ticket.get('body', '')}"
    embedding = embedder.embed_text(text)  # 384-dim

    # Repo one-hot
    repo_onehot = [1.0 if ticket["repo"] == R else 0.0 for R in REPOS]

    # Label flags
    labels = ticket.get("labels", "") or ""
    has_bug         = 1.0 if "bug" in labels else 0.0
    has_enhancement = 1.0 if "enhancement" in labels else 0.0
    has_crash       = 1.0 if "crash" in labels else 0.0

    # Numeric features
    comments  = float(ticket.get("comments_count") or 0)
    seniority = float(ticket.get("seniority_enum") or 0)
    hist_avg  = float(ticket.get("historical_avg_completion_hours") or 0)
    kw_count  = float(len(ticket.get("keywords") or []))

    # Text length
    title_len = float(len(ticket.get("title") or ""))
    body_len  = float(len(ticket.get("body") or ""))

    engineered = repo_onehot + [has_bug, has_enhancement, has_crash,
                                comments, seniority, hist_avg, kw_count,
                                title_len, body_len]

    features = np.hstack([embedding, engineered]).astype(np.float32)
    return np.nan_to_num(features, nan=0.0).reshape(1, -1)


def predict(ticket: dict) -> float:
    """Predict completion time in business hours."""
    features = build_features(ticket)
    log_pred = model.predict(features)[0]
    return round(np.expm1(log_pred), 1)


# ── Sample tickets ───────────────────────────────────────────────────────────
test_tickets = [
    {
        "title": "Terraform crashes with nil pointer dereference on plan",
        "body": "Running terraform plan against an S3 backend causes a nil pointer panic. Reproducible on v1.6.0. Stack trace attached.",
        "repo": "hashicorp/terraform",
        "labels": "bug",
        "comments_count": 3,
        "seniority_enum": 1,
        "historical_avg_completion_hours": 48,
        "keywords": ["terraform", "aws", "s3", "go"],
    },
    {
        "title": "Add support for dynamic inventory caching in Ansible",
        "body": "Currently dynamic inventory scripts are called on every run. Adding a TTL-based cache would significantly speed up large playbook runs.",
        "repo": "ansible/ansible",
        "labels": "enhancement",
        "comments_count": 7,
        "seniority_enum": 1,
        "historical_avg_completion_hours": 72,
        "keywords": ["ansible", "python", "inventory", "cache"],
    },
    {
        "title": "Prometheus query returns incorrect results for histogram_quantile",
        "body": "histogram_quantile returns NaN for certain label combinations when the bucket series have gaps. Happens under high cardinality.",
        "repo": "prometheus/prometheus",
        "labels": "bug",
        "comments_count": 12,
        "seniority_enum": 2,
        "historical_avg_completion_hours": 30,
        "keywords": ["prometheus", "go", "histogram", "metrics"],
    },
    {
        "title": "Simple typo fix in Terraform documentation",
        "body": "Fix typo in resource docs for aws_instance.",
        "repo": "hashicorp/terraform",
        "labels": "",
        "comments_count": 1,
        "seniority_enum": 0,
        "historical_avg_completion_hours": 5,
        "keywords": ["terraform", "docs"],
    },
]

# ── Run predictions ──────────────────────────────────────────────────────────
print(f"{'Ticket':<55} {'Repo':<25} {'Labels':<15} {'Predicted hrs':>13}")
print("-" * 115)
for t in test_tickets:
    pred = predict(t)
    title = t["title"][:52] + "..." if len(t["title"]) > 52 else t["title"]
    print(f"{title:<55} {t['repo']:<25} {t.get('labels',''):<15} {pred:>10.1f} hrs")