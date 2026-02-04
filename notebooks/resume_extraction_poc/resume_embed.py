"""
Step 3: Resume Embedder
-----------------------
Creates vector embeddings from normalized resume text.

Uses sentence-transformers (local, free).

Install:
    pip install sentence-transformers
"""

from typing import List, Dict
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddedResume:
    """Resume with embedding."""
    engineer_id: str
    filename: str
    normalized_content: str
    embedding: List[float]


class ResumeEmbedder:
    """Generate embeddings for resume text."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedder with specified model.

        Args:
            model_name: sentence-transformers model name
                       Options: 'all-MiniLM-L6-v2' (384 dim, fast)
                               'all-mpnet-base-v2' (768 dim, balanced)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"    ✓ Loaded. Embedding dimension: {self.embedding_dim}")

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Normalized resume text

        Returns:
            List of floats (embedding vector)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, normalized_resumes: List, show_progress: bool = True) -> List[EmbeddedResume]:
        """
        Generate embeddings for multiple resumes.

        Args:
            normalized_resumes: List of NormalizedResume objects from Step 2
            show_progress: Show progress bar

        Returns:
            List of EmbeddedResume objects
        """
        print(f"\nGenerating embeddings for {len(normalized_resumes)} resumes...")

        # Extract texts for batch encoding
        texts = [r.normalized_content for r in normalized_resumes]

        # Batch encode for efficiency
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )

        # Create EmbeddedResume objects
        results = []
        for resume, embedding in zip(normalized_resumes, embeddings):
            embedded = EmbeddedResume(
                engineer_id=resume.engineer_id,
                filename=resume.filename,
                normalized_content=resume.normalized_content,
                embedding=embedding.tolist()
            )
            results.append(embedded)

        print(f"    ✓ Generated {len(results)} embeddings ({self.embedding_dim} dimensions each)")

        return results

    def get_model_info(self) -> Dict:
        """Get embedding model information."""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
        }