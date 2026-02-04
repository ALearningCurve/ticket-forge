"""Step 4: Resume Storage - Stores final processed resumes to JSON and SQLite."""

import json
import sqlite3
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class ResumeRecord:
    engineer_id: str
    filename: str
    normalized_content: str
    embedding: List[float]
    embedding_model: str
    embedding_dim: int
    processed_at: str


class ResumeStorage:
    def __init__(self, embedded_resumes: List, embedding_model: str, embedding_dim: int):
        self.records: List[ResumeRecord] = []
        timestamp = datetime.utcnow().isoformat()

        for resume in embedded_resumes:
            self.records.append(ResumeRecord(
                engineer_id=resume.engineer_id,
                filename=resume.filename,
                normalized_content=resume.normalized_content,
                embedding=resume.embedding,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                processed_at=timestamp
            ))

    def save_json(self, output_path: str):
        data = [asdict(r) for r in self.records]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_sqlite(self, output_path: str):
        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()

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

        for record in self.records:
            cursor.execute('''
                INSERT OR REPLACE INTO resumes 
                (engineer_id, filename, normalized_content, embedding, embedding_model, embedding_dim, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.engineer_id,
                record.filename,
                record.normalized_content,
                json.dumps(record.embedding),
                record.embedding_model,
                record.embedding_dim,
                record.processed_at
            ))

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_engineer_id ON resumes(engineer_id)')
        conn.commit()
        conn.close()

    def save(self, json_path: str, sqlite_path: str):
        self.save_json(json_path)
        self.save_sqlite(sqlite_path)


class ResumeReader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_all(self) -> List[Dict]:
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM resumes')
        count = cursor.fetchone()[0]
        conn.close()
        return count