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
    """Represents an engineer's cold-start profile derived from a resume."""

    engineer_id: str
    github_username: Optional[str]
    full_name: Optional[str]
    embedding: List[float]
    keywords: List[str]
    confidence: float
    experience_weight: float
    created_at: str


class ColdStartManager:
    """Manages cold-start profile creation from resumes and persistence to Postgres."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        default_confidence: float = 0.3,
        default_experience_weight: float = 0.3,
    ) -> None:
        """Initialize the manager with a Postgres DSN and embedding config."""
        self.dsn = dsn or os.environ.get("DATABASE_URL")
        if not self.dsn:
            msg = "No Postgres DSN provided. Pass `dsn` or set the DATABASE_URL env var."
            raise RuntimeError(msg)

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
        self,
        file_path: str,
        github_username: Optional[str] = None,
        full_name: Optional[str] = None,
    ) -> EngineerProfile:
        """Extract, normalize, and embed a single resume into an EngineerProfile."""
        extracted = self.extractor.extract(str(file_path))
        text = (
            extracted
            if isinstance(extracted, str)
            else getattr(extracted, "raw_content", "")
        )

        normalized_text, _ = self.normalizer.normalize(text)
        keywords = self.keyword_extractor.extract(normalized_text)
        emb = self.embed_service.embed_text(normalized_text)

        return EngineerProfile(
            engineer_id=(
                getattr(extracted, "engineer_id", None) or Path(file_path).stem
            ),
            github_username=github_username,
            full_name=full_name or github_username or Path(file_path).stem,
            embedding=emb.tolist() if hasattr(emb, "tolist") else list(map(float, emb)),
            keywords=keywords,
            confidence=self.default_confidence,
            experience_weight=self.default_experience_weight,
            created_at=datetime.utcnow().isoformat(),
        )

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
            gh_user = None
            key = f.stem
            if username_map and key in username_map:
                gh_user = username_map[key]
            try:
                p = self.process_resume_file(str(f), github_username=gh_user)
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

    def save_profile(self, profile: EngineerProfile) -> dict:
        """Save a single profile. Returns member_id and action."""
        return self._upsert_profiles([profile])[0]

    def save_profiles(self, profiles: Iterable[EngineerProfile]) -> List[dict]:
        """Save multiple profiles into the Postgres `users` table."""
        return self._upsert_profiles(list(profiles))

    def _upsert_profiles(self, profiles: List[EngineerProfile]) -> List[dict]:
        """Upsert profiles using github_username as the lookup key.

        Expects the pgvector extension enabled and the schema from
        `scripts/postgres/init/02_schema.sql` applied.
        """
        conn = self._get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        results = []

        try:
            for p in profiles:
                vec_text = "[" + ",".join(map(str, p.embedding)) + "]"
                keywords_text = " ".join(p.keywords) if p.keywords else ""

                # Lookup by github_username first, fall back to full_name
                row = None
                if p.github_username:
                    cur.execute(
                        "SELECT member_id FROM users WHERE github_username = %s",
                        (p.github_username,),
                    )
                    row = cur.fetchone()

                if not row and p.full_name:
                    cur.execute(
                        "SELECT member_id FROM users WHERE full_name = %s",
                        (p.full_name,),
                    )
                    row = cur.fetchone()

                if row:
                    cur.execute(
                        """
                        UPDATE users SET
                          github_username    = COALESCE(%s, github_username),
                          full_name          = COALESCE(%s, full_name),
                          resume_base_vector = %s::vector,
                          profile_vector     = %s::vector,
                          skill_keywords     = to_tsvector('english', %s),
                          confidence         = %s,
                          experience_weight  = %s,
                          updated_at         = now()
                        WHERE member_id = %s
                        RETURNING member_id
                        """,
                        (
                            p.github_username,
                            p.full_name,
                            vec_text,
                            vec_text,
                            keywords_text,
                            p.confidence,
                            p.experience_weight,
                            row["member_id"],
                        ),
                    )
                    result = cur.fetchone()
                    results.append({
                        "member_id": str(result["member_id"]),
                        "action": "updated",
                    })
                else:
                    cur.execute(
                        """
                        INSERT INTO users
                          (github_username, full_name,
                           resume_base_vector, profile_vector,
                           skill_keywords, confidence,
                           experience_weight)
                        VALUES
                          (%s, %s, %s::vector, %s::vector,
                           to_tsvector('english', %s), %s, %s)
                        RETURNING member_id
                        """,
                        (
                            p.github_username,
                            p.full_name,
                            vec_text,
                            vec_text,
                            keywords_text,
                            p.confidence,
                            p.experience_weight,
                        ),
                    )
                    result = cur.fetchone()
                    results.append({
                        "member_id": str(result["member_id"]),
                        "action": "created",
                    })

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

        return results


# ---------------------------------------------------------------------- #
#  Convenience runner
# ---------------------------------------------------------------------- #

def run_coldstart(resume_dir: str, dsn: Optional[str] = None) -> None:
    """Process all resumes in a directory and save profiles to Postgres."""
    mgr = ColdStartManager(dsn=dsn)
    profiles = mgr.process_directory(resume_dir)
    results = mgr.save_profiles(profiles)
    print(f"Saved {len(results)} profile(s) to Postgres.")
    for r in results:
        print(f"  {r['member_id']} → {r['action']}")


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