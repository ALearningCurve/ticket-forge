"""Cold-start profile initializer from resumes.

This module builds an initial engineer profile from a resume by:
- extracting text (OCR for PDF),
- normalizing text (remove PII),
- extracting skill keywords using `ml_core.keywords`,
- generating an embedding using `ml_core.embeddings`,
- assigning a conservative confidence/experience weight,
- persisting the profile to a local SQLite DB for testing and providing
  an example Postgres INSERT statement for production.

The implementation intentionally keeps DB write logic simple so it can be
reused in CI/local testing. For production, use the Postgres scripts under
`scripts/postgres/init` and a proper DB connection pool.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ml_core.embeddings import get_embedding_service
from ml_core.keywords import get_keyword_extractor

from training.training.etl.ingest.resume.resume_extract import ResumeExtractor
from training.training.etl.ingest.resume.resume_normalize import ResumeNormalizer


@dataclass
class EngineerProfile:
    engineer_id: str
    username: Optional[str]
    embedding: List[float]
    keywords: List[str]
    confidence: float
    experience_weight: float
    created_at: str


class ColdStartManager:
    def __init__(
        self,
        db_path: str = "profiles.db",
        embedding_model: str = "all-MiniLM-L6-v2",
        default_confidence: float = 0.3,
        default_experience_weight: float = 0.3,
    ) -> None:
        self.db_path = db_path
        self.extractor = ResumeExtractor()
        self.normalizer = ResumeNormalizer()
        self.keyword_extractor = get_keyword_extractor()
        self.embed_service = get_embedding_service(model_name=embedding_model)
        self.default_confidence = default_confidence
        self.default_experience_weight = default_experience_weight

    def process_resume_file(self, file_path: str, username: Optional[str] = None) -> EngineerProfile:
        extracted = self.extractor.extract(str(file_path))
        # extractor.extract returns string for single-file extractor in this module
        # but ResumeExtractor.extract_directory returns objects; handle both
        text = extracted if isinstance(extracted, str) else getattr(extracted, "raw_content", "")

        normalized_text, _ = self.normalizer.normalize(text)

        keywords = self.keyword_extractor.extract(normalized_text)

        emb = self.embed_service.embed_text(normalized_text)

        profile = EngineerProfile(
            engineer_id=(getattr(extracted, "engineer_id", None) or Path(file_path).stem),
            username=username,
            embedding=emb.tolist() if hasattr(emb, "tolist") else list(map(float, emb)),
            keywords=keywords,
            confidence=self.default_confidence,
            experience_weight=self.default_experience_weight,
            created_at=datetime.utcnow().isoformat(),
        )

        return profile

    def process_directory(self, resume_dir: str, username_map: Optional[Dict[str, str]] = None) -> List[EngineerProfile]:
        """Process all resumes in a directory.

        `username_map` is an optional mapping from filename (without suffix) to github username.
        """
        dir_path = Path(resume_dir)
        profiles: List[EngineerProfile] = []

        files = [p for p in dir_path.iterdir() if p.is_file()]
        for f in files:
            uname = None
            key = f.stem
            if username_map and key in username_map:
                uname = username_map[key]
            try:
                p = self.process_resume_file(str(f), username=uname)
                profiles.append(p)
            except Exception:
                # skip failures for now; caller may log
                continue

        return profiles

    def ensure_sqlite_schema(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS engineers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engineer_id TEXT UNIQUE,
                username TEXT,
                embedding TEXT,
                keywords TEXT,
                confidence REAL,
                experience_weight REAL,
                created_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def save_profiles_sqlite(self, profiles: Iterable[EngineerProfile]) -> None:
        self.ensure_sqlite_schema()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        for p in profiles:
            cur.execute(
                """
                INSERT INTO engineers (engineer_id, username, embedding, keywords, confidence, experience_weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(engineer_id) DO UPDATE SET
                  username=excluded.username,
                  embedding=excluded.embedding,
                  keywords=excluded.keywords,
                  confidence=excluded.confidence,
                  experience_weight=excluded.experience_weight,
                  created_at=excluded.created_at
                """,
                (
                    p.engineer_id,
                    p.username,
                    json.dumps(p.embedding),
                    json.dumps(p.keywords),
                    p.confidence,
                    p.experience_weight,
                    p.created_at,
                ),
            )

        conn.commit()
        conn.close()

    def save_profiles_postgres(self, profiles: Iterable[EngineerProfile], dsn: Optional[str] = None) -> None:
        """Save profiles into Postgres `users` table defined in scripts/postgres/init/02_schema.sql.

        Expects a running Postgres with the pgvector extension and the schema applied.
        Connection is taken from `dsn` or `DATABASE_URL` env var.
        """
        try:
            import os

            import psycopg2
            from psycopg2.extras import RealDictCursor
        except Exception as e:
            raise RuntimeError("psycopg2 is required to save to Postgres: pip install psycopg2-binary") from e

        conn_str = dsn or os.environ.get("DATABASE_URL")
        if not conn_str:
            raise RuntimeError("No Postgres DSN provided; set DATABASE_URL or pass dsn")

        conn = psycopg2.connect(conn_str)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        for p in profiles:
            # Try to find existing user by full_name (best-effort match)
            full_name = p.username or p.engineer_id
            cur.execute("SELECT member_id FROM users WHERE full_name = %s", (full_name,))
            row = cur.fetchone()

            vec_text = "[" + ",".join(map(str, p.embedding)) + "]"
            keywords_text = " ".join(p.keywords) if p.keywords else ""

            if row:
                # Update existing profile: update resume_base_vector and profile_vector
                cur.execute(
                    """
                    UPDATE users SET
                      resume_base_vector = %s::vector,
                      profile_vector = %s::vector,
                      skill_keywords = to_tsvector('english', %s),
                      updated_at = now()
                    WHERE member_id = %s
                    """,
                    (vec_text, vec_text, keywords_text, row["member_id"]),
                )
            else:
                # Insert new user; tickets_closed_count defaults to 0
                cur.execute(
                    """
                    INSERT INTO users (full_name, resume_base_vector, profile_vector, skill_keywords)
                    VALUES (%s, %s::vector, %s::vector, to_tsvector('english', %s))
                    RETURNING member_id
                    """,
                    (full_name, vec_text, vec_text, keywords_text),
                )

        conn.commit()
        cur.close()
        conn.close()

    @staticmethod
    def postgres_insert_example(profile: EngineerProfile, table: str = "engineers") -> str:
        """Return an example parameterized INSERT for Postgres with pgvector.

        Note: This is a template. Use a proper DB client + parameterization.
        """
        # Example assumes a pgvector column `embedding_vector` of type vector
        insert = (
            "INSERT INTO {table} (engineer_id, username, embedding, keywords, confidence, experience_weight, created_at)"
            " VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6, $7)"
            " ON CONFLICT (engineer_id) DO UPDATE SET username = EXCLUDED.username;"
        ).format(table=table)

        return insert


def run_coldstart_from_dir(resume_dir: str, db_path: str = "profiles.db") -> None:
    mgr = ColdStartManager(db_path=db_path)
    profiles = mgr.process_directory(resume_dir)
    mgr.save_profiles_sqlite(profiles)


def run_coldstart_postgres(resume_dir: str, dsn: Optional[str] = None) -> None:
    mgr = ColdStartManager()
    profiles = mgr.process_directory(resume_dir)
    mgr.save_profiles_postgres(profiles, dsn=dsn)


print("hello")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cold-start engineer profiles from resumes")
    parser.add_argument("resume_dir", help="Directory with resume files")
    parser.add_argument("--db", default="profiles.db", help="SQLite DB path to store profiles")
    parser.add_argument("--pg", action="store_true", help="Save profiles to Postgres instead of SQLite")
    parser.add_argument("--dsn", default=None, help="Postgres DSN or DATABASE_URL to use when --pg is set")

    args = parser.parse_args()
    if args.pg:
        run_coldstart_postgres(args.resume_dir, dsn=args.dsn)
    else:
        run_coldstart_from_dir(args.resume_dir, db_path=args.db)
