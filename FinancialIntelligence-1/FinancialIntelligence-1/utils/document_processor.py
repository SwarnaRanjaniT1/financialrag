import os
import re
import pandas as pd
import numpy as np
from io import StringIO

# For PDF processing
try:
    import PyPDF2
except ImportError:
    # Fallback
    import pdfplumber

def process_document(file_path):
    """Process a document and split it into chunks for embedding.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        list: List of text chunks from the document
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            return process_pdf(file_path)
        elif file_extension == '.csv':
            return process_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return process_excel(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return []
        
def process_pdf(file_path, chunk_size=1000, overlap=200):
    """Process a PDF file and split it into overlapping chunks.
    
    Args:
        file_path (str): Path to the PDF file
        chunk_size (int): Maximum size of each chunk
        overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks from the PDF
    """
    extracted_text = ""
    
    try:
        # Try PyPDF2 first
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text() + "\n\n"
    except:
        # Fallback to pdfplumber if PyPDF2 fails
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted_text += page.extract_text() + "\n\n"
        except ImportError:
            raise ImportError("Neither PyPDF2 nor pdfplumber is available. Install one of them.")
    
    # Clean up the text
    extracted_text = clean_text(extracted_text)
    
    # Split into chunks
    chunks = []
    current_pos = 0
    
    while current_pos < len(extracted_text):
        # Get chunk with specified size
        chunk_end = min(current_pos + chunk_size, len(extracted_text))
        
        # Try to end at a sentence or paragraph
        if chunk_end < len(extracted_text):
            # Look for paragraph end
            paragraph_end = extracted_text.rfind('\n\n', current_pos, chunk_end)
            if paragraph_end != -1 and paragraph_end > current_pos + chunk_size / 2:
                chunk_end = paragraph_end + 2
            else:
                # Look for sentence end
                sentence_end = max(
                    extracted_text.rfind('. ', current_pos, chunk_end),
                    extracted_text.rfind('? ', current_pos, chunk_end),
                    extracted_text.rfind('! ', current_pos, chunk_end)
                )
                if sentence_end != -1 and sentence_end > current_pos + chunk_size / 2:
                    chunk_end = sentence_end + 2
        
        # Add the chunk
        chunk = extracted_text[current_pos:chunk_end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move position with overlap
        current_pos = max(current_pos + chunk_size - overlap, current_pos + 1)
        if current_pos >= len(extracted_text):
            break
    
    return chunks

def process_csv(file_path, max_rows_per_chunk=50):
    """Process a CSV file and convert it into text chunks.
    
    Args:
        file_path (str): Path to the CSV file
        max_rows_per_chunk (int): Maximum number of rows per chunk
        
    Returns:
        list: List of text chunks representing the CSV data
    """
    try:
        df = pd.read_csv(file_path)
        return dataframe_to_chunks(df, max_rows_per_chunk)
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return []

def process_excel(file_path, max_rows_per_chunk=50):
    """Process an Excel file and convert it into text chunks.
    
    Args:
        file_path (str): Path to the Excel file
        max_rows_per_chunk (int): Maximum number of rows per chunk
        
    Returns:
        list: List of text chunks representing the Excel data
    """
    try:
        xlsx = pd.ExcelFile(file_path)
        chunks = []
        
        # Process each sheet in the Excel file
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            sheet_chunks = dataframe_to_chunks(df, max_rows_per_chunk, sheet_name)
            chunks.extend(sheet_chunks)
        
        return chunks
    except Exception as e:
        print(f"Error processing Excel: {str(e)}")
        return []

def dataframe_to_chunks(df, max_rows_per_chunk=50, sheet_name=None):
    """Convert a DataFrame to text chunks.
    
    Args:
        df (pandas.DataFrame): DataFrame to convert
        max_rows_per_chunk (int): Maximum number of rows per chunk
        sheet_name (str, optional): Name of the sheet if from Excel
        
    Returns:
        list: List of text chunks
    """
    chunks = []
    total_rows = len(df)
    
    # Replace NaN values with empty strings
    df = df.fillna('')
    
    # Get column names and datatypes for schema information
    columns = df.columns.tolist()
    datatypes = df.dtypes.to_dict()
    
    # Format datatypes as strings
    datatype_info = {col: str(dtype) for col, dtype in datatypes.items()}
    
    # Create a schema information chunk
    schema_info = "Schema Information:\n"
    if sheet_name:
        schema_info += f"Sheet: {sheet_name}\n"
    schema_info += f"Total Rows: {total_rows}\n"
    schema_info += "Columns:\n"
    
    for col in columns:
        schema_info += f"- {col} ({datatype_info[col]})\n"
    
    chunks.append(schema_info)
    
    # Process data in chunks
    for i in range(0, total_rows, max_rows_per_chunk):
        end_idx = min(i + max_rows_per_chunk, total_rows)
        chunk_df = df.iloc[i:end_idx]
        
        # Convert chunk to string
        buffer = StringIO()
        chunk_df.to_csv(buffer, index=False)
        csv_str = buffer.getvalue()
        
        # Add metadata to the chunk
        chunk_text = f"Data Rows {i+1} to {end_idx}"
        if sheet_name:
            chunk_text += f" (Sheet: {sheet_name})"
        chunk_text += ":\n" + csv_str
        
        chunks.append(chunk_text)
    
    return chunks

def clean_text(text):
    """Clean and normalize text.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Normalize whitespace around punctuation
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
    text = re.sub(r'([({])\s+', r'\1', text)
    
    return text.strip()
