import streamlit as st
import os
import tempfile
from utils.document_processor import process_document
from utils.embeddings import generate_embeddings
from utils.retriever import retrieve_context
from utils.generator import generate_answer
from utils.guardrails import input_validator, output_validator

# Set page configuration
st.set_page_config(
    page_title="Financial Q&A Assistant",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = ""
if 'context_used' not in st.session_state:
    st.session_state.context_used = []

# Title and description
st.title("Financial Statement Q&A Assistant")
st.markdown("""
This application allows you to upload financial statements and ask questions about them.
It uses a custom Retrieval-Augmented Generation (RAG) system to provide relevant answers from your financial documents.
""")

# Sidebar
with st.sidebar:
    st.header("Upload Financial Documents")
    st.markdown("Supported formats: PDF, CSV, Excel")
    
    uploaded_file = st.file_uploader("Choose a financial document", 
                                     type=["pdf", "csv", "xlsx", "xls"],
                                     help="Upload your financial statements here")
    
    # Process the uploaded file
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            
            # Process the document based on its type
            chunks = process_document(temp_file_path)
            
            if chunks:
                # Generate embeddings for the document chunks
                embeddings = generate_embeddings(chunks)
                
                # Store in session state
                st.session_state.document_chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.file_uploaded = True
                
                st.success(f"Document processed successfully! {len(chunks)} chunks generated.")
                
                # Clean up the temporary file
                os.unlink(temp_file_path)
            else:
                st.error("Failed to process the document. Please try a different file.")
    
    # Display information about the processed document
    if st.session_state.file_uploaded:
        st.info(f"Document processed with {len(st.session_state.document_chunks)} chunks.")
        
        if st.button("Clear Document"):
            st.session_state.document_chunks = []
            st.session_state.embeddings = None
            st.session_state.file_uploaded = False
            st.session_state.last_question = ""
            st.session_state.last_answer = ""
            st.session_state.context_used = []
            st.experimental_rerun()

# Main area
if st.session_state.file_uploaded:
    st.header("Ask Questions About Your Financial Document")
    
    # Input for user question
    question = st.text_input("Enter your financial question:",
                            help="Ask any question related to the uploaded financial document",
                            placeholder="E.g., What is the company's revenue for 2022?")
    
    # Process the question when the user submits it
    if question and question != st.session_state.last_question:
        # Apply input guardrails
        is_valid, message = input_validator(question)
        
        if is_valid:
            with st.spinner("Generating answer..."):
                # Retrieve relevant context using RAG
                context, context_indices = retrieve_context(
                    question,
                    st.session_state.document_chunks,
                    st.session_state.embeddings
                )
                
                # Generate answer using the model
                answer = generate_answer(question, context)
                
                # Apply output guardrails
                answer = output_validator(answer)
                
                # Store in session state
                st.session_state.last_question = question
                st.session_state.last_answer = answer
                st.session_state.context_used = [st.session_state.document_chunks[i] for i in context_indices]
                
        else:
            st.error(message)
    
    # Display the answer if available
    if st.session_state.last_answer:
        st.subheader("Answer:")
        st.markdown(st.session_state.last_answer)
        
        # Option to show context used
        with st.expander("Show source context"):
            for i, ctx in enumerate(st.session_state.context_used):
                st.markdown(f"**Context {i+1}:**")
                st.markdown(f"```\n{ctx}\n```")
                st.markdown("---")
else:
    # If no document is uploaded yet
    st.info("👈 Please upload a financial document first using the sidebar.")
    
    # Example questions
    st.subheader("Example questions you can ask:")
    example_questions = [
        "What was the total revenue in the last fiscal year?",
        "How did the company's gross profit margin change year-over-year?",
        "What are the largest expense categories?",
        "How much cash does the company have on hand?",
        "What is the company's debt-to-equity ratio?"
    ]
    
    for q in example_questions:
        st.markdown(f"- {q}")

# Footer
st.markdown("---")
st.markdown("This application was built using Streamlit and a custom Retrieval-Augmented Generation (RAG) implementation.")
