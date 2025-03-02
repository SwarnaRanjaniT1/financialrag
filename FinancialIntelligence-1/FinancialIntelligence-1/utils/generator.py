import os
from utils.retriever import augment_query_with_context

def generate_answer(query, contexts, max_new_tokens=512, temperature=0.7):
    """Generate an answer based on the retrieved contexts using a simple text-based approach.
    
    Since we don't have access to the DeepSeek R1 Qwen 1.5B model, this function creates
    a response based on the retrieved context chunks.
    
    Args:
        query (str): User query
        contexts (list): List of relevant context chunks
        max_new_tokens (int): Maximum number of tokens to generate (not used in this implementation)
        temperature (float): Generation temperature (not used in this implementation)
        
    Returns:
        str: Generated answer
    """
    try:
        if not contexts or len(contexts) == 0:
            return "I don't have sufficient information in the document to answer this question."
        
        # Create a basic answer from the contexts
        answer = ""
        
        # Get the query keywords (simple implementation)
        query_lower = query.lower()
        
        # List of words to ignore in keyword extraction
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'for', 'to', 'and', 
                    'with', 'on', 'by', 'from', 'about', 'as', 'how', 'much', 'many', 'where', 
                    'when', 'which', 'who', 'whom', 'whose', 'why', 'can', 'could', 'did', 
                    'do', 'does', 'has', 'have', 'had', 'been', 'being', 'was', 'were', 'will'}
        
        # Extract keywords from query
        keywords = [word for word in query_lower.split() if word not in stopwords]
        
        # Generate a response based on the retrieved contexts and keywords
        relevant_sentences = []
        
        for context in contexts:
            context_lower = context.lower()
            
            # Check if the context contains any of the keywords
            if any(keyword in context_lower for keyword in keywords):
                # Split the context into sentences (simple implementation)
                sentences = context.split('. ')
                
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in keywords):
                        if sentence not in relevant_sentences and len(sentence) > 10:
                            relevant_sentences.append(sentence)
        
        # If we found relevant sentences, use them to build the answer
        if relevant_sentences:
            answer = "Based on the financial document, "
            
            # Add the relevant sentences to the answer
            for i, sentence in enumerate(relevant_sentences[:5]):  # Limit to 5 sentences
                if i > 0:
                    answer += " "
                    
                # Clean the sentence
                clean_sentence = sentence.strip()
                if not clean_sentence.endswith('.'):
                    clean_sentence += '.'
                    
                answer += clean_sentence
        else:
            # If no relevant sentences found, use a generic response with the first context
            answer = "I found the following information in the document that might help answer your question: " + contexts[0]
            
        return answer
    
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return f"Sorry, I encountered an error while generating the answer. Error details: {str(e)}"

def load_model():
    """Placeholder for model loading - we're not loading an actual model in this implementation."""
    return True, True

def clear_model():
    """Placeholder for model clearing - we're not loading an actual model in this implementation."""
    pass
