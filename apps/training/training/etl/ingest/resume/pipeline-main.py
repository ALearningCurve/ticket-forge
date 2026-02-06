"""Resume Pipeline - Main Runner"""

from pathlib import Path
import json

from shared.configuration import Paths
from resume_extract import ResumeExtractor
from resume_normalize import ResumeNormalizer
from resume_embed import ResumeEmbedder
from resume_store import ResumeStorage, ResumeReader


def run_pipeline(
    resume_directory: str,
    output_dir: str = "output",
    output_json: str = "resumes_final.json",
    output_sqlite: str = "resumes_final.db",
    embedding_model: str = "all-MiniLM-L6-v2",
    id_prefix: str = "ENG",
    use_uuid: bool = False
):
    input_dir = Path(resume_directory)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {resume_directory}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_json = str(output_path / output_json)
    output_sqlite = str(output_path / output_sqlite)

    # Step 1: Extract
    extractor = ResumeExtractor(id_prefix=id_prefix, use_uuid=use_uuid)
    extracted_resumes = extractor.extract_directory(resume_directory)

    if not extracted_resumes:
        raise ValueError("No resumes extracted")

    # Step 2: Normalize
    normalizer = ResumeNormalizer(
        remove_phone=True,
        remove_email=True,
        remove_url=True,
        remove_address=True,
        remove_dates=True,
        remove_gpa=True,
    )
    normalized_resumes = normalizer.normalize_batch(extracted_resumes)

    # Step 3: Embed
    embedder = ResumeEmbedder(model_name=embedding_model)
    embedded_resumes = embedder.embed_batch(normalized_resumes)
    model_info = embedder.get_model_info()

    # Step 4: Store
    storage = ResumeStorage(
        embedded_resumes=embedded_resumes,
        embedding_model=model_info['model_name'],
        embedding_dim=model_info['embedding_dim']
    )
    storage.save(json_path=output_json, sqlite_path=output_sqlite)

    return ResumeReader(output_sqlite).count()


if __name__ == "__main__":
    RESUME_DIRECTORY = r"./data/team_resume"


    OUTPUT_DIR = Paths.data_root / r"apps/training/training/etl/ingest/resume/output"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    OUTPUT_JSON = "resumes_final.json"
    OUTPUT_SQLITE = "resumes_final.db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    ID_PREFIX = "ENG"
    USE_UUID = False

    run_pipeline(
        resume_directory=RESUME_DIRECTORY,
        output_dir=OUTPUT_DIR,
        output_json=OUTPUT_JSON,
        output_sqlite=OUTPUT_SQLITE,
        embedding_model=EMBEDDING_MODEL,
        id_prefix=ID_PREFIX,
        use_uuid=USE_UUID
    )

    reader = ResumeReader(str(OUTPUT_DIR / OUTPUT_SQLITE))

    for r in reader.get_all():
        print(json.dumps({
            'engineer_id': r['engineer_id'],
            'filename': r['filename'],
            'normalized_content': r['normalized_content'][:200] + '...',
            'embedding': r['embedding'],
            'embedding_model': r['embedding_model']
        }, indent=2))
        print("-" * 50)