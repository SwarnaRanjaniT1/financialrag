import numpy as np
from utils.embeddings import embed_query, cosine_similarity

def retrieve_context(query, documents, embeddings, top_k=3, similarity_threshold=0.2):
    """Retrieve relevant document chunks for a query.
    
    Args:
        query (str): User query
        documents (list): List of document chunks
        embeddings (numpy.ndarray): Precomputed embeddings for document chunks
        top_k (int): Number of top chunks to retrieve
        similarity_threshold (float): Minimum similarity score to consider
        
    Returns:
        tuple: (list of contexts, list of context indices)
    """
    if not documents or embeddings is None or len(documents) == 0:
        return [], []
    
    # Embed the query
    query_embedding = embed_query(query)
    
    # Calculate similarities
    similarities = []
    for i, doc_embedding in enumerate(embeddings):
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Filter by threshold and get top_k
    filtered_similarities = [(i, score) for i, score in similarities if score >= similarity_threshold]
    top_similarities = filtered_similarities[:top_k]
    
    # Extract contexts and their indices
    context_indices = [i for i, _ in top_similarities]
    contexts = [documents[i] for i in context_indices]
    
    return contexts, context_indices

def rerank_contexts(query, contexts, context_indices):
    """Rerank contexts based on semantic similarity to the query.
    
    This function can be used for more sophisticated reranking if needed.
    
    Args:
        query (str): User query
        contexts (list): List of context chunks
        context_indices (list): Original indices of the contexts
        
    Returns:
        tuple: (reranked contexts, reranked context indices)
    """
    # For now, we'll just return the contexts in their original order
    # This function can be expanded later with more advanced reranking logic
    return contexts, context_indices

def augment_query_with_context(query, contexts, max_context_length=2000):
    """Augment the query with retrieved contexts for the generator.
    
    Args:
        query (str): User query
        contexts (list): Retrieved context chunks
        max_context_length (int): Maximum length of combined context
        
    Returns:
        str: Augmented query with context
    """
    combined_context = ""
    total_length = 0
    
    for i, context in enumerate(contexts):
        # Add context with a marker for its origin
        context_addition = f"\nContext {i+1}:\n{context}\n"
        
        # Check if adding this context would exceed the max length
        if total_length + len(context_addition) > max_context_length:
            # If first context is already too long, truncate it
            if i == 0:
                context_addition = context_addition[:max_context_length]
                combined_context += context_addition
            break
        
        combined_context += context_addition
        total_length += len(context_addition)
    
    # Create the augmented query
    augmented_query = f"""Answer the following question based on the provided context. If you cannot find information in the context to answer the question, say "I don't have sufficient information to answer this question."

Context:
{combined_context}

Question: {query}

Answer:"""
    
    return augmented_query
