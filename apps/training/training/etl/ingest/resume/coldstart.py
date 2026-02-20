"""Cold-start profile initializer from resumes.

This module builds an initial engineer profile from a resume by:
- extracting text (OCR for PDF),
- normalizing text (remove PII),
- extracting skill keywords using `ml_core.keywords`,
- generating an embedding using `ml_core.embeddings`,
- assigning a conservative confidence/experience weight,
- persisting the profile to a Postgres DB with pgvector.

Requires a running Postgres instance with the pgvector extension and
the schema from `scripts/postgres/init/02_schema.sql` applied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from ml_core.embeddings import get_embedding_service
from ml_core.keywords import get_keyword_extractor

from training.etl.ingest.resume.resume_extract import ResumeExtractor
from training.etl.ingest.resume.resume_normalize import ResumeNormalizer


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
        dsn: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        default_confidence: float = 0.3,
        default_experience_weight: float = 0.3,
    ) -> None:
        self.dsn = dsn or os.environ.get("DATABASE_URL")
        if not self.dsn:
            raise RuntimeError(
                "No Postgres DSN provided. Pass `dsn` or set the DATABASE_URL env var."
            )

        self.extractor = ResumeExtractor()
        self.normalizer = ResumeNormalizer()
        self.keyword_extractor = get_keyword_extractor()
        self.embed_service = get_embedding_service(model_name=embedding_model)
        self.default_confidence = default_confidence
        self.default_experience_weight = default_experience_weight

    # ------------------------------------------------------------------ #
    #  Resume processing
    # ------------------------------------------------------------------ #

    def process_resume_file(
        self, file_path: str, username: Optional[str] = None
    ) -> EngineerProfile:
        extracted = self.extractor.extract(str(file_path))
        text = (
            extracted
            if isinstance(extracted, str)
            else getattr(extracted, "raw_content", "")
        )

        normalized_text, _ = self.normalizer.normalize(text)
        keywords = self.keyword_extractor.extract(normalized_text)
        emb = self.embed_service.embed_text(normalized_text)

        profile = EngineerProfile(
            engineer_id=(
                getattr(extracted, "engineer_id", None) or Path(file_path).stem
            ),
            username=username,
            embedding=emb.tolist() if hasattr(emb, "tolist") else list(map(float, emb)),
            keywords=keywords,
            confidence=self.default_confidence,
            experience_weight=self.default_experience_weight,
            created_at=datetime.utcnow().isoformat(),
        )

        return profile

    def process_directory(
        self,
        resume_dir: str,
        username_map: Optional[Dict[str, str]] = None,
    ) -> List[EngineerProfile]:
        """Process all resumes in a directory.

        `username_map` is an optional mapping from filename (without suffix)
        to github username.
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

    # ------------------------------------------------------------------ #
    #  Postgres persistence
    # ------------------------------------------------------------------ #

    def _get_connection(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(self.dsn)

    def save_profiles(self, profiles: Iterable[EngineerProfile]) -> None:
        """Save profiles into the Postgres `users` table.

        Expects the pgvector extension enabled and the schema from
        `scripts/postgres/init/02_schema.sql` applied.
        """
        conn = self._get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            for p in profiles:
                full_name = p.username or p.engineer_id
                cur.execute(
                    "SELECT member_id FROM users WHERE full_name = %s",
                    (full_name,),
                )
                row = cur.fetchone()

                vec_text = "[" + ",".join(map(str, p.embedding)) + "]"
                keywords_text = " ".join(p.keywords) if p.keywords else ""

                if row:
                    cur.execute(
                        """
                        UPDATE users SET
                          resume_base_vector = %s::vector,
                          profile_vector     = %s::vector,
                          skill_keywords     = to_tsvector('english', %s),
                          updated_at         = now()
                        WHERE member_id = %s
                        """,
                        (vec_text, vec_text, keywords_text, row["member_id"]),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO users
                          (full_name, resume_base_vector, profile_vector, skill_keywords)
                        VALUES
                          (%s, %s::vector, %s::vector, to_tsvector('english', %s))
                        RETURNING member_id
                        """,
                        (full_name, vec_text, vec_text, keywords_text),
                    )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()


# ---------------------------------------------------------------------- #
#  Convenience runner
# ---------------------------------------------------------------------- #

def run_coldstart(resume_dir: str, dsn: Optional[str] = None) -> None:
    mgr = ColdStartManager(dsn=dsn)
    profiles = mgr.process_directory(resume_dir)
    mgr.save_profiles(profiles)
    print(f"Saved {len(profiles)} profile(s) to Postgres.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cold-start engineer profiles from resumes into Postgres"
    )
    parser.add_argument("resume_dir", help="Directory with resume files")
    parser.add_argument(
        "--dsn",
        default=None,
        help="Postgres DSN (defaults to DATABASE_URL env var)",
    )

    args = parser.parse_args()
    run_coldstart(args.resume_dir, dsn=args.dsn)