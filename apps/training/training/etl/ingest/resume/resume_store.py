"""Step 4: Resume Storage - Stores final processed resumes to JSON and SQLite."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ResumeRecord:
  """Represents a complete resume record with embeddings."""

  engineer_id: str
  filename: str
  normalized_content: str
  embedding: List[float]
  embedding_model: str
  embedding_dim: int
  processed_at: str


class ResumeStorage:
  """Stores processed resumes to JSON and SQLite databases."""

  def __init__(
    self,
    embedded_resumes: List[Any],
    embedding_model: str,
    embedding_dim: int,
  ) -> None:
    """Initialize storage with embedded resumes.

    Args:
      embedded_resumes: List of embedded resume objects.
      embedding_model: Name of the embedding model used.
      embedding_dim: Dimension of the embedding vectors.
    """
    self.records: List[ResumeRecord] = []
    timestamp = datetime.utcnow().isoformat()

    for resume in embedded_resumes:
      self.records.append(
        ResumeRecord(
          engineer_id=resume.engineer_id,
          filename=resume.filename,
          normalized_content=resume.normalized_content,
          embedding=resume.embedding,
          embedding_model=embedding_model,
          embedding_dim=embedding_dim,
          processed_at=timestamp,
        )
      )

  def save_json(self, output_path: str) -> None:
    """Save resumes to JSON file.

    Args:
      output_path: Path to output JSON file.
    """
    data = [asdict(r) for r in self.records]
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(data, f, indent=2, ensure_ascii=False)

  def save_sqlite(self, output_path: str) -> None:
    """Save resumes to SQLite database.

    Args:
      output_path: Path to output SQLite database file.
    """
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    cursor.execute("""
      CREATE TABLE IF NOT EXISTS resumes (
        engineer_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        normalized_content TEXT NOT NULL,
        embedding BLOB NOT NULL,
        embedding_model TEXT NOT NULL,
        embedding_dim INTEGER NOT NULL,
        processed_at TEXT NOT NULL
      )
    """)

    for record in self.records:
      cursor.execute(
        """
        INSERT OR REPLACE INTO resumes 
        (engineer_id, filename, normalized_content, embedding, 
         embedding_model, embedding_dim, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      """,
        (
          record.engineer_id,
          record.filename,
          record.normalized_content,
          json.dumps(record.embedding),
          record.embedding_model,
          record.embedding_dim,
          record.processed_at,
        ),
      )

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_engineer_id ON resumes(engineer_id)")
    conn.commit()
    conn.close()

  def save(self, json_path: str, sqlite_path: str) -> None:
    """Save resumes to both JSON and SQLite.

    Args:
      json_path: Path to output JSON file.
      sqlite_path: Path to output SQLite database file.
    """
    self.save_json(json_path)
    self.save_sqlite(sqlite_path)


class ResumeReader:
  """Reads resume data from SQLite database."""

  def __init__(self, db_path: str) -> None:
    """Initialize reader with database path.

    Args:
      db_path: Path to SQLite database file.
    """
    self.db_path = db_path

  def get_all(self) -> List[Dict[str, Any]]:
    """Get all resumes from database.

    Returns:
      List of resume dictionaries.
    """
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM resumes")
    rows = cursor.fetchall()

    results = []
    for row in rows:
      record = dict(row)
      record["embedding"] = json.loads(record["embedding"])
      results.append(record)

    conn.close()
    return results

  def get_by_id(self, engineer_id: str) -> Optional[Dict[str, Any]]:
    """Get resume by engineer ID.

    Args:
      engineer_id: Engineer ID to search for.

    Returns:
      Resume dictionary if found, None otherwise.
    """
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM resumes WHERE engineer_id = ?", (engineer_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
      record = dict(row)
      record["embedding"] = json.loads(record["embedding"])
      return record
    return None

  def get_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
    """Get resume by filename.

    Args:
      filename: Filename to search for.

    Returns:
      Resume dictionary if found, None otherwise.
    """
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM resumes WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    conn.close()

    if row:
      record = dict(row)
      record["embedding"] = json.loads(record["embedding"])
      return record
    return None

  def count(self) -> int:
    """Get total number of resumes in database.

    Returns:
      Number of resumes.
    """
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM resumes")
    count = cursor.fetchone()[0]
    conn.close()
    return count
