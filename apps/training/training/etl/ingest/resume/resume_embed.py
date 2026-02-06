"""Step 3: Resume Embedder - Creates vector embeddings from normalized resume text."""

from dataclasses import dataclass
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddedResume:
  """Represents a resume with its embedding vector."""

  engineer_id: str
  filename: str
  normalized_content: str
  embedding: List[float]


class ResumeEmbedder:
  """Generates vector embeddings from resume text using sentence transformers."""

  def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
    """Initialize the embedder with a sentence transformer model.

    Args:
      model_name: Name of the sentence transformer model to use.
    """
    self.model = SentenceTransformer(model_name)
    self.model_name = model_name
    self.embedding_dim = self.model.get_sentence_embedding_dimension()

  def embed(self, text: str) -> List[float]:
    """Generate embedding vector for a single text.

    Args:
      text: Text to embed.

    Returns:
      Embedding vector as list of floats.
    """
    embedding = self.model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

  def embed_batch(
    self,
    normalized_resumes: List[Any],
    show_progress: bool = True,
  ) -> List[EmbeddedResume]:
    """Generate embeddings for a batch of resumes.

    Args:
      normalized_resumes: List of normalized resume objects.
      show_progress: Whether to show progress bar.

    Returns:
      List of embedded resume objects.
    """
    texts = [r.normalized_content for r in normalized_resumes]

    embeddings = self.model.encode(
      texts,
      convert_to_numpy=True,
      show_progress_bar=show_progress,
    )

    results = []
    for resume, embedding in zip(normalized_resumes, embeddings, strict=True):
      results.append(
        EmbeddedResume(
          engineer_id=resume.engineer_id,
          filename=resume.filename,
          normalized_content=resume.normalized_content,
          embedding=embedding.tolist(),
        )
      )

    return results

  def get_model_info(self) -> Dict[str, Any]:
    """Get information about the embedding model.

    Returns:
      Dictionary with model name and embedding dimension.
    """
    return {
      "model_name": self.model_name,
      "embedding_dim": self.embedding_dim,
    }