"""
Resume Pipeline - Main Runner
=============================
Connects all 4 steps: Extract → Normalize → Embed → Store

Install dependencies:
    pip install pymupdf python-docx sentence-transformers

Usage:
    python pipeline_main.py
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
    embedding_model: str = "all-MiniLM-L6-v2",
    id_prefix: str = "ENG",
    use_uuid: bool = False
):
    """
    Run the complete resume processing pipeline.

    Args:
        resume_directory: Path to folder with PDF/DOCX/DOC files
        output_json: Output JSON file path
        output_sqlite: Output SQLite database path
        embedding_model: sentence-transformers model name
        id_prefix: Prefix for engineer IDs (e.g., "ENG" → ENG-001)
        use_uuid: Use UUID instead of sequential IDs
    """

    print("=" * 60)
    print("RESUME PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input Directory: {resume_directory}")
    print(f"Output JSON:     {output_json}")
    print(f"Output SQLite:   {output_sqlite}")
    print(f"Embedding Model: {embedding_model}")
    print(f"ID Format:       {'UUID' if use_uuid else f'{id_prefix}-001, {id_prefix}-002, ...'}")
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

    extractor = ResumeExtractor(id_prefix=id_prefix, use_uuid=use_uuid)
    extracted_resumes = extractor.extract_directory(resume_directory)
    # extracted_resumes = [ExtractedResume(engineer_id, filename, raw_content), ...]

    if not extracted_resumes:
        print("❌ No resumes extracted. Exiting.")
        return

    print(f"\n✓ Step 1 Complete: Extracted {len(extracted_resumes)} resumes")

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

    normalized_resumes = normalizer.normalize_batch(extracted_resumes)
    # normalized_resumes = [NormalizedResume(engineer_id, filename, normalized_content, removed_items), ...]

    print(f"\n✓ Step 2 Complete: Normalized {len(normalized_resumes)} resumes")

    # ─────────────────────────────────────────────────────────
    # STEP 3: EMBED
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 3: GENERATING EMBEDDINGS")
    print("─" * 60)

    embedder = ResumeEmbedder(model_name=embedding_model)
    embedded_resumes = embedder.embed_batch(normalized_resumes)
    # embedded_resumes = [EmbeddedResume(engineer_id, filename, normalized_content, embedding), ...]

    model_info = embedder.get_model_info()

    print(f"\n✓ Step 3 Complete: Generated {len(embedded_resumes)} embeddings")

    # ─────────────────────────────────────────────────────────
    # STEP 4: STORE
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 4: SAVING TO FINAL STORAGE")
    print("─" * 60)

    storage = ResumeStorage(
        embedded_resumes=embedded_resumes,
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

    reader = ResumeReader(output_sqlite)
    records = reader.get_all()
    total_count = reader.count()

    print(f"\n📊 DATABASE SUMMARY:")
    print("-" * 60)
    print(f"Total Records:       {total_count}")
    print(f"Embedding Model:     {model_info['model_name']}")
    print(f"Embedding Dimension: {model_info['embedding_dim']}")

    # ─────────────────────────────────────────────────────────
    # DISPLAY ALL RECORDS FROM DATABASE
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📋 ALL RECORDS IN DATABASE:")
    print("=" * 60)

    for i, record in enumerate(records, 1):
        print(f"\n[{i}] Engineer ID: {record['engineer_id']}")
        print(f"    Filename:       {record['filename']}")
        print(f"    Content Length: {len(record['normalized_content'])} chars")
        print(f"    Content Preview: {record['normalized_content'][:80]}...")
        print(f"    Embedding:      [{record['embedding'][0]:.4f}, {record['embedding'][1]:.4f}, ...] (dim={len(record['embedding'])})")

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

    # ID generation options:
    # - ID_PREFIX = "ENG" → ENG-001, ENG-002, ENG-003, ...
    # - USE_UUID = True   → A1B2C3D4, E5F6G7H8, ...
    ID_PREFIX = "ENG"
    USE_UUID = False

    # ══════════════════════════════════════════════════════════

    run_pipeline(
        resume_directory=RESUME_DIRECTORY,
        output_json=OUTPUT_JSON,
        output_sqlite=OUTPUT_SQLITE,
        embedding_model=EMBEDDING_MODEL,
        id_prefix=ID_PREFIX,
        use_uuid=USE_UUID
    )