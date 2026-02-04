"""
Step 4: Resume Storage
----------------------
Stores final processed resumes (normalized content + embeddings).

Supports:
- JSON (for review/portability)
- SQLite (for querying)
"""

import json
import sqlite3
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class ResumeRecord:
    """Final resume record schema."""
    engineer_id: str
    filename: str
    normalized_content: str
    embedding: List[float]
    embedding_model: str
    embedding_dim: int
    processed_at: str


class ResumeStorage:
    """Store processed resumes to JSON and SQLite."""

    def __init__(
        self,
        embedded_resumes: List,
        embedding_model: str,
        embedding_dim: int
    ):
        """
        Initialize storage with processed data.

        Args:
            embedded_resumes: List of EmbeddedResume objects from Step 3
            embedding_model: Name of embedding model used
            embedding_dim: Dimension of embeddings
        """
        self.records: List[ResumeRecord] = []

        timestamp = datetime.utcnow().isoformat()

        for resume in embedded_resumes:
            record = ResumeRecord(
                engineer_id=resume.engineer_id,
                filename=resume.filename,
                normalized_content=resume.normalized_content,
                embedding=resume.embedding,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                processed_at=timestamp
            )
            self.records.append(record)

    def save_json(self, output_path: str):
        """
        Save to JSON file.

        Args:
            output_path: Path for JSON output
        """
        data = [asdict(r) for r in self.records]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(self.records)} records to JSON: {output_path}")

    def save_sqlite(self, output_path: str):
        """
        Save to SQLite database.

        Args:
            output_path: Path for SQLite database
        """
        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()

        # Create table with filename column
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                engineer_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                normalized_content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                processed_at TEXT NOT NULL
            )
        ''')

        # Insert records
        for record in self.records:
            embedding_json = json.dumps(record.embedding)

            cursor.execute('''
                INSERT OR REPLACE INTO resumes 
                (engineer_id, filename, normalized_content, embedding, embedding_model, embedding_dim, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.engineer_id,
                record.filename,
                record.normalized_content,
                embedding_json,
                record.embedding_model,
                record.embedding_dim,
                record.processed_at
            ))

        # Create index
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_engineer_id ON resumes(engineer_id)')

        conn.commit()
        conn.close()

        print(f"✅ Saved {len(self.records)} records to SQLite: {output_path}")

    def save(self, json_path: str, sqlite_path: str):
        """Save to both JSON and SQLite."""
        self.save_json(json_path)
        self.save_sqlite(sqlite_path)


class ResumeReader:
    """Read stored resumes from SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_all(self) -> List[Dict]:
        """Get all resume records."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM resumes')
        rows = cursor.fetchall()

        results = []
        for row in rows:
            record = dict(row)
            record['embedding'] = json.loads(record['embedding'])
            results.append(record)

        conn.close()
        return results

    def get_by_id(self, engineer_id: str) -> Optional[Dict]:
        """Get resume by engineer ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM resumes WHERE engineer_id = ?', (engineer_id,))
        row = cursor.fetchone()

        conn.close()

        if row:
            record = dict(row)
            record['embedding'] = json.loads(record['embedding'])
            return record
        return None

    def get_by_filename(self, filename: str) -> Optional[Dict]:
        """Get resume by original filename."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM resumes WHERE filename = ?', (filename,))
        row = cursor.fetchone()

        conn.close()

        if row:
            record = dict(row)
            record['embedding'] = json.loads(record['embedding'])
            return record
        return None

    def count(self) -> int:
        """Get total number of records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM resumes')
        count = cursor.fetchone()[0]
        conn.close()
        return count