"""
Resume Pipeline - Main Runner
=============================
Connects all 4 steps: Extract → Normalize → Embed → Store

Install dependencies:
    pip install pymupdf python-docx sentence-transformers

Usage:
    python pipeline_main.py

File structure required:
    your_project/
    ├── resume_extract.py
    ├── resume_normalize.py
    ├── resume_embed.py
    ├── resume_store.py
    ├── pipeline_main.py      <-- RUN THIS
    └── /resume/              <-- Your PDF/DOCX files
"""

from pathlib import Path

# Import all pipeline modules
from resume_extract import ResumeExtractor
from resume_normalize import ResumeNormalizer
from resume_embed import ResumeEmbedder
from resume_store import ResumeStorage, ResumeReader


def run_pipeline(
    resume_directory: str,
    output_json: str = "resumes_final.json",
    output_sqlite: str = "resumes_final.db",
    embedding_model: str = "all-MiniLM-L6-v2"
):
    """
    Run the complete resume processing pipeline.

    Data Flow:
        resume files (PDF/DOCX/DOC)
            │
            ▼ Step 1: Extract
        raw_data: Dict[engineer_id, raw_text]
            │
            ▼ Step 2: Normalize
        normalized_data: Dict[engineer_id, clean_text]
            │
            ▼ Step 3: Embed
        embeddings: Dict[engineer_id, vector]
            │
            ▼ Step 4: Store
        JSON + SQLite files
    """

    print("=" * 60)
    print("RESUME PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input Directory: {resume_directory}")
    print(f"Output JSON:     {output_json}")
    print(f"Output SQLite:   {output_sqlite}")
    print(f"Embedding Model: {embedding_model}")
    print("=" * 60)

    # Validate input directory
    input_dir = Path(resume_directory)
    if not input_dir.exists():
        print(f"❌ Error: Directory not found: {resume_directory}")
        return

    # ─────────────────────────────────────────────────────────
    # STEP 1: EXTRACT
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 1: EXTRACTING TEXT FROM RESUMES")
    print("─" * 60)

    extractor = ResumeExtractor()
    raw_data = extractor.extract_directory(resume_directory)

    if not raw_data:
        print("❌ No resumes extracted. Exiting.")
        return

    print(f"\n✓ Step 1 Complete: Extracted {len(raw_data)} resumes")

    # ─────────────────────────────────────────────────────────
    # STEP 2: NORMALIZE
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 2: NORMALIZING (REMOVING PII, DATES, ETC.)")
    print("─" * 60)

    normalizer = ResumeNormalizer(
        remove_phone=True,
        remove_email=True,
        remove_url=True,
        remove_address=True,
        remove_dates=True,
        remove_gpa=True,
    )

    normalized_data, removed_data = normalizer.normalize_batch(raw_data)

    print(f"\n✓ Step 2 Complete: Normalized {len(normalized_data)} resumes")

    # ─────────────────────────────────────────────────────────
    # STEP 3: EMBED
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 3: GENERATING EMBEDDINGS")
    print("─" * 60)

    embedder = ResumeEmbedder(model_name=embedding_model)
    embeddings = embedder.embed_batch(normalized_data)

    model_info = embedder.get_model_info()

    print(f"\n✓ Step 3 Complete: Generated {len(embeddings)} embeddings")

    # ─────────────────────────────────────────────────────────
    # STEP 4: STORE
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 4: SAVING TO FINAL STORAGE")
    print("─" * 60)

    storage = ResumeStorage(
        normalized_data=normalized_data,
        embeddings=embeddings,
        embedding_model=model_info['model_name'],
        embedding_dim=model_info['embedding_dim']
    )

    storage.save(json_path=output_json, sqlite_path=output_sqlite)

    # ─────────────────────────────────────────────────────────
    # VERIFY & DISPLAY FROM DATABASE
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE - VERIFYING DATABASE")
    print("=" * 60)

    # Read from database to verify
    reader = ResumeReader(output_sqlite)
    records = reader.get_all()
    total_count = reader.count()

    print(f"\n📊 DATABASE SUMMARY:")
    print("-" * 60)
    print(f"Total Records in DB: {total_count}")
    print(f"Output JSON:         {output_json}")
    print(f"Output SQLite:       {output_sqlite}")

    if records:
        # Get info from first record (actual data from DB)
        first_record = records[0]
        print(f"\nEmbedding Model:     {first_record['embedding_model']}")
        print(f"Embedding Dimension: {first_record['embedding_dim']}")
        print(f"Processed At:        {first_record['processed_at']}")

    # ─────────────────────────────────────────────────────────
    # DISPLAY ALL RECORDS FROM DATABASE
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📋 ALL RECORDS IN DATABASE:")
    print("=" * 60)

    for i, record in enumerate(records, 1):
        print(f"\n[{i}] Engineer ID: {record['engineer_id']}")
        print(f"    Content Length: {len(record['normalized_content'])} chars")
        print(f"    Content Preview: {record['normalized_content'][:100]}...")
        print(f"    Embedding: [{record['embedding'][0]:.4f}, {record['embedding'][1]:.4f}, {record['embedding'][2]:.4f}, ...] (dim={len(record['embedding'])})")

    print("\n" + "=" * 60)
    print(f"✅ Total: {total_count} resumes processed and stored")
    print("=" * 60)


# ════════════════════════════════════════════════════════════
# CONFIGURATION & RUN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ══════════════════════════════════════════════════════════
    # EDIT THESE VALUES
    # ══════════════════════════════════════════════════════════

    RESUME_DIRECTORY = r"D:\Ticketforge-MLOps\ticket-forge\notebooks\resume_extraction_poc\test_resume"

    OUTPUT_JSON = "resumes_final.json"
    OUTPUT_SQLITE = "resumes_final.db"

    # Embedding model options:
    # - 'all-MiniLM-L6-v2'     (384 dim, fast)
    # - 'all-mpnet-base-v2'    (768 dim, more accurate)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # ══════════════════════════════════════════════════════════

    run_pipeline(
        resume_directory=RESUME_DIRECTORY,
        output_json=OUTPUT_JSON,
        output_sqlite=OUTPUT_SQLITE,
        embedding_model=EMBEDDING_MODEL
    )