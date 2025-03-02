import re

def input_validator(query):
    """Validate user input to prevent harmful queries.
    
    Args:
        query (str): User query to validate
        
    Returns:
        tuple: (is_valid, message)
    """
    # Check if query is empty or too short
    if not query or len(query.strip()) < 3:
        return False, "Please enter a valid question with at least 3 characters."
    
    # Maximum allowed query length
    if len(query) > 500:
        return False, "Your question is too long. Please limit it to 500 characters."
    
    # Check for potentially sensitive requests
    sensitive_patterns = [
        r"\bpassword(s)?\b",
        r"\bcredit\s*card(s)?\b",
        r"\bsocial\s*security\b",
        r"\bhack(ing)?\b",
        r"\bexploit\b",
        r"\battack\b",
        r"\billegal\b",
        r"\bunethical\b"
    ]
    
    for pattern in sensitive_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "Your question appears to be requesting sensitive or potentially harmful information."
    
    # Check specifically for financial document analysis appropriateness
    financial_query = check_financial_relevance(query)
    if not financial_query:
        return True, ""  # Still allow but with warning in output validator
    
    return True, ""

def output_validator(answer):
    """Validate model output to ensure safe and appropriate responses.
    
    Args:
        answer (str): Generated answer to validate
        
    Returns:
        str: Validated (possibly modified) answer
    """
    # Check for empty or too short answers
    if not answer or len(answer.strip()) < 10:
        return "I couldn't generate a meaningful answer to your question. Please try rephrasing or ask a different question."
    
    # Check for harmful content
    harmful_patterns = [
        r"\bhow\s+to\s+hack\b",
        r"\bhow\s+to\s+exploit\b",
        r"\bhow\s+to\s+attack\b",
        r"\billegal\s+\w+\s+to\b",
        r"\bunethical\s+\w+\s+to\b"
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return "I apologize, but I cannot provide information that could be used for harmful purposes."
    
    # Check for disclaimers that might need to be added
    if "financial advice" in answer.lower() or "investment advice" in answer.lower():
        disclaimer = "\n\n**Disclaimer**: This response is for informational purposes only and does not constitute financial advice. Always consult with a qualified financial professional before making investment decisions."
        if disclaimer not in answer:
            answer += disclaimer
    
    # Check if the answer contains uncertainty markers but doesn't explicitly state uncertainty
    uncertainty_markers = ["might", "may", "could", "possibly", "perhaps", "I think", "likely"]
    uncertainty_count = sum(1 for marker in uncertainty_markers if marker in answer.lower())
    
    explicit_uncertainty = "I'm not entirely sure" in answer or "I'm uncertain" in answer
    
    if uncertainty_count >= 2 and not explicit_uncertainty:
        answer = "Note: There is some uncertainty in this answer. Please verify with the source document.\n\n" + answer
    
    # Remove this check as it's causing all answers to be marked as not financially relevant
    # This is because we're now looking at the answer text, not the original query
    # if not check_financial_relevance(answer):
    #     answer = "This question may not be directly related to financial statements. The answer provided is based on my general knowledge.\n\n" + answer
    
    return answer

def check_financial_relevance(text):
    """Check if the text is relevant to financial analysis.
    
    Args:
        text (str): Text to check
        
    Returns:
        bool: True if financially relevant, False otherwise
    """
    financial_terms = [
        r"\bbalance\s*sheet\b",
        r"\bincome\s*statement\b",
        r"\bcash\s*flow\b",
        r"\bfinancial\s*statement\b",
        r"\bassets?\b",
        r"\bliabilit(y|ies)\b",
        r"\bequity\b",
        r"\brevenue\b",
        r"\bprofit\b",
        r"\bloss\b",
        r"\bexpenses?\b",
        r"\btaxes?\b",
        r"\bdividend\b",
        r"\bearnings\b",
        r"\bebitda\b",
        r"\bratio\b",
        r"\bmargin\b",
        r"\bdebt\b",
        r"\bq[1-4]\b",
        r"\bfiscal\b",
        r"\bquarter(ly)?\b",
        r"\bannual\b"
    ]
    
    for pattern in financial_terms:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False
