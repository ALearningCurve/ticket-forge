"""Step 3: Resume Embedder - Creates vector embeddings from normalized resume text."""

from typing import List, Dict
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddedResume:
    engineer_id: str
    filename: str
    normalized_content: str
    embedding: List[float]


class ResumeEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, normalized_resumes: List, show_progress: bool = True) -> List[EmbeddedResume]:
        texts = [r.normalized_content for r in normalized_resumes]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )

        results = []
        for resume, embedding in zip(normalized_resumes, embeddings):
            results.append(EmbeddedResume(
                engineer_id=resume.engineer_id,
                filename=resume.filename,
                normalized_content=resume.normalized_content,
                embedding=embedding.tolist()
            ))

        return results

    def get_model_info(self) -> Dict:
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
        }