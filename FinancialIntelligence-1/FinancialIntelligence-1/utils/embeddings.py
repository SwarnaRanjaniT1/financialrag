import numpy as np
from sentence_transformers import SentenceTransformer
import os

def generate_embeddings(text_chunks, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for a list of text chunks.
    
    Args:
        text_chunks (list): List of text chunks to embed
        model_name (str): Name of the sentence transformer model to use
        
    Returns:
        numpy.ndarray: Array of embeddings for each chunk
    """
    try:
        # Load the model (cache if already loaded)
        model = SentenceTransformer(model_name)
        
        # Generate embeddings
        embeddings = model.encode(text_chunks, show_progress_bar=True)
        
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        # Return empty array with appropriate dimensions
        # Get the dimension from model if possible, otherwise use default
        try:
            dim = model.get_sentence_embedding_dimension()
        except:
            dim = 384  # Default for all-MiniLM-L6-v2
        return np.zeros((len(text_chunks), dim))

def embed_query(query, model_name="all-MiniLM-L6-v2"):
    """Generate embedding for a single query.
    
    Args:
        query (str): Query string to embed
        model_name (str): Name of the sentence transformer model to use
        
    Returns:
        numpy.ndarray: Embedding vector for the query
    """
    try:
        # Load the model (cache if already loaded)
        model = SentenceTransformer(model_name)
        
        # Generate embedding
        embedding = model.encode(query)
        
        return embedding
    except Exception as e:
        print(f"Error embedding query: {str(e)}")
        # Return empty array with appropriate dimensions
        try:
            dim = model.get_sentence_embedding_dimension()
        except:
            dim = 384  # Default for all-MiniLM-L6-v2
        return np.zeros(dim)

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors.
    
    Args:
        v1 (numpy.ndarray): First vector
        v2 (numpy.ndarray): Second vector
        
    Returns:
        float: Cosine similarity score
    """
    # Handle zero vectors
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)
