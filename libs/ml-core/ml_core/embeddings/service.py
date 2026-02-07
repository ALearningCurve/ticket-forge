"""
Embedding service for converting text to semantic vectors.
"""
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service to convert text data into high-dimensional vectors using pre-trained models.
    
    This ensures tickets and engineer profiles are embedded in the same vector space,
    enabling meaningful semantic similarity comparisons.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a single text string into a semantic vector.
        
        Args:
            text: Cleaned text string from ticket or engineer profile
            
        Returns:
            384-dimensional dense vector representing semantic meaning
            
        Raises:
            ValueError: If text is empty or None
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding
    
    def embed_batch(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed multiple texts efficiently in batches.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show a progress bar
            
        Returns:
            Array of embeddings with shape (n_texts, 384)
            
        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        return self.embedding_dim


# Singleton instance for reuse across the application
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(
    model_name: str = 'all-MiniLM-L6-v2',
    force_reload: bool = False
) -> EmbeddingService:
    """
    Get or create the global embedding service instance.
    
    Args:
        model_name: Model to use (only applies on first call)
        force_reload: Force reloading the model
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None or force_reload:
        _embedding_service = EmbeddingService(model_name=model_name)
    
    return _embedding_service