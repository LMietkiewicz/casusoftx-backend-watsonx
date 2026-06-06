"""processing.py - core processing logic for PDF text and table extraction, chunking, and embedding preparation"""
import re
from typing import List, Dict, Any, Tuple
import json
from sentence_transformers import SentenceTransformer

# --- Part 1: Text and Table Extraction ---

def extract_text_and_tables(doc: object) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts all text and tables from a PDF.
    
    Args:
        pdf_path (str): The file path to the PDF document.
        
    Returns:
        Tuple[str, List[Dict[str, Any]]]: A tuple containing:
            - A single string of all concatenated text.
            - A list of dictionaries, where each dict represents a structured table.
    """

    full_text = ""
    tables = []
    
    for page_num, page in enumerate(doc):
        # Extract text
        full_text += page.get_text("text") + "\n\n"
        
        # Extract tables
        page_tables = page.find_tables()
        for table in page_tables:
            table_data = table.extract()
            if table_data:
                tables.append({
                    "page_number": page_num + 1,
                    "data": table_data
                })
                
    return full_text.strip(), tables

# --- Part 2: Recursive Chunker for Parent Chunks ---

def create_text_parent_chunks(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Recursively splits text to create larger, overlapping "parent" chunks.
    This is the same recursive chunker as before.
    
    Args:
        text (str): The input text to be chunked.
        separators (List[str]): A list of separators to split the text by.
        chunk_size (int): The maximum size of each parent chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        
    Returns:
        List[str]: A list of parent text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    # Try splitting with the highest-priority separator
    current_separator = separators[0]
    
    if not current_separator or current_separator not in text:
        # If separator not found or we're at the end of our separator list,
        # try the next one.
        if len(separators) > 1:
            return create_text_parent_chunks(text, separators[1:], chunk_size, chunk_overlap)
        else:
            # If no separators left, force split.
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunks.append(text[i:i + chunk_size])
            return chunks

    # Split by the current separator
    parts = text.split(current_separator)
    final_chunks = []
    current_chunk = ""
    
    for part in parts:
        # If adding the next part doesn't exceed the chunk size, add it.
        if len(current_chunk) + len(part) + len(current_separator) <= chunk_size:
            current_chunk += part + current_separator
        else:
            # If the current chunk is not empty, it's a complete chunk.
            if current_chunk:
                final_chunks.append(current_chunk.strip())
            
            # If the part itself is too large, recurse on it.
            if len(part) > chunk_size:
                sub_chunks = create_text_parent_chunks(part, separators[1:], chunk_size, chunk_overlap)
                final_chunks.extend(sub_chunks)
            else:
                # Start a new chunk with the current part
                current_chunk = part + current_separator

    # Add the last remaining chunk
    if current_chunk:
        final_chunks.append(current_chunk.strip())
        
    return final_chunks

# --- Part 3: Parent-Child Chunking Logic (Sentence-Aware) ---

def create_text_child_chunks(parent_chunks: List[str], child_chunk_size: int) -> List[Dict[str, str]]:
    """
    Splits parent chunks into smaller, semantically meaningful child chunks.
    This version splits by sentences and then groups them into chunks that
    are less than the child_chunk_size, ensuring no sentences are broken.
    
    Args:
        parent_chunks (List[str]): The list of larger, overlapping chunks.
        child_chunk_size (int): The target maximum size of the smaller child chunks.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary
                                contains a 'child' chunk and its corresponding 'parent'.
    """
    chunk_pairs = []
    for i, parent_chunk in enumerate(parent_chunks):
        parent_id = i
        # If the parent is small enough, it is its own child.
        if len(parent_chunk) <= child_chunk_size:
            chunk_pairs.append({'child_contents': parent_chunk, 'parent_contents': parent_chunk, 'parent_id': parent_id})
            continue

        # This complex regex splits on sentence endings, but uses negative lookbehinds
        # to avoid splitting on abbreviations (e.g., Mr.), initials (e.g., A. B.),
        # and enumerations (e.g., 1. or a.).
        # It also splits on multiple newlines (paragraph breaks).
        sentence_split_regex = r'(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<!\b[a-z]\.)(?<!\b\d\.)(?<!\b\d\d\.)(?<!\b\d\d\d\.)(?<=[.!?])\s+|\n\n+'
        sentences = re.split(sentence_split_regex, parent_chunk)
        
        current_child_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding the next sentence doesn't exceed the size, append it.
            if len(current_child_chunk) + len(sentence) + 1 <= child_chunk_size:
                current_child_chunk += sentence + " "
            else:
                # If the current child chunk is not empty, save it.
                if current_child_chunk:
                    chunk_pairs.append({'child_contents': current_child_chunk.strip(), 'parent_id': parent_id, 'parent_contents': parent_chunk, 'type': 'text'})
                
                # If the sentence itself is larger than the chunk size, it becomes its own chunk.
                if len(sentence) > child_chunk_size:
                    chunk_pairs.append({'child_contents': sentence, 'parent_id': parent_id, 'parent_contents': parent_chunk, 'type': 'text'})
                    current_child_chunk = "" # Reset
                else:
                    # Otherwise, start a new child chunk with the current sentence.
                    current_child_chunk = sentence + " "

        # Add the last remaining child chunk if it exists.
        if current_child_chunk:
            chunk_pairs.append({'child_contents': current_child_chunk.strip(), 'parent_id': parent_id, 'parent_contents': parent_chunk, 'type': 'text'})
            
    return chunk_pairs

def create_table_chunks(tables: List[Dict[str, Any]], parent_id_start_index: int = 0) -> List[Dict[str, Any]]:
    """
    Creates parent-child chunks from structured table data.
    Parent: The full table as a compact JSON string.
    Child: A descriptive sentence for each row.
    """
    chunk_pairs = []
    for i, table_obj in enumerate(tables):
        parent_id = parent_id_start_index + i
        table_data = table_obj.get('data')
        
        # Robustness Check 1: Ensure table_data is a list with at least a header.
        if not isinstance(table_data, list) or not table_data:
            continue
            
        # Robustness Check 2: Filter out "junk" tables with meaningless headers.
        header_cells = [str(cell).strip() for cell in table_data[0] if cell is not None and str(cell).strip()]
        if len(header_cells) < 2 and len(table_data) < 2: # Require at least 2 header columns or 2 rows
            continue

        # --- New Logic: Create the parent chunk as a JSON string ---
        header = [str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in table_data[0]]
        data_rows = table_data[1:] if len(table_data) > 1 else []
        
        list_of_dicts = []
        for row in data_rows:
            # Normalize cell content and create a dictionary for the row
            row_dict = {}
            for j, cell in enumerate(row):
                col_name = header[j] if j < len(header) else f"column_{j+1}"
                cell_value = str(cell).replace('\n', ' ').strip() if cell is not None else ""
                if col_name: # Only add if the column has a name
                    row_dict[col_name] = cell_value
            if row_dict:
                list_of_dicts.append(row_dict)
        
        # Convert the list of dictionaries to a compact JSON string
        parent_json = json.dumps(list_of_dicts, ensure_ascii=False, separators=(',', ':'))
        
        # Create child chunks (one per row)
        for row in data_rows:
            child_sentence = "W tej tabeli wiersz zawiera następujące dane:"
            row_clauses = []
            for j, cell in enumerate(row):
                col_name = header[j] if j < len(header) else ""
                cell_value = str(cell).replace('\n', ' ').strip() if cell is not None else ""
                
                if col_name and cell_value:
                    row_clauses.append(f"wartość dla '{col_name}' to '{cell_value}'")
            
            if row_clauses:
                child_sentence += "; ".join(row_clauses) + "."
                chunk_pairs.append({'child_contents': child_sentence, 'parent_id': parent_id, 'parent_contents': parent_json, 'type': 'table'})
            
    return chunk_pairs

# --- Part 4: Full Processing Pipeline ---

def processing_pipeline(doc: object, file_id: str, model: SentenceTransformer) -> List[Dict[str, Any]]:
    """
    The main pipeline function that processes a PDF from start to finish.
    """
    # 1. Extraction
    full_text, tables = extract_text_and_tables(doc)

    # 2. Text Chunking
    text_parent_chunks = create_text_parent_chunks(full_text, ["\n\n", "\n", ". "], 1500, 200)
    text_chunk_pairs = create_text_child_chunks(text_parent_chunks, 400)
    
    # 3. Table Chunking
    num_text_parents = len(text_parent_chunks)
    table_chunk_pairs = create_table_chunks(tables, num_text_parents)

    # 4. Data Transformation and Embedding
    final_chunks = []
    processed_parents = {}
    
    # Get embedding dimension for placeholder vectors
    #model = SentenceTransformer('sdadas/mmlw-retrieval-roberta-large-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    parent_placeholder_dense = [0.0] * embedding_dim
    
    # Collect all child contents for efficient batch embedding
    child_contents_to_embed = []
    
    # Process text chunks
    for pair in text_chunk_pairs:
        parent_id = pair['parent_id']
        child_contents_to_embed.append(pair['child_contents'])
        if parent_id not in processed_parents:
            final_chunks.append({
                "file_id": file_id,
                "parent_id": parent_id,
                "hierarchy": "parent",
                "type": "text",
                "contents": pair['parent_contents'],
                "dense_embedding": parent_placeholder_dense
            })
            processed_parents[parent_id] = True

    # Process table chunks
    for pair in table_chunk_pairs:
        # Create a globally unique parent_id
        parent_id = pair['parent_id']
        child_contents_to_embed.append(pair['child_contents'])
        if parent_id not in processed_parents:
            final_chunks.append({
                "file_id": file_id,
                "parent_id": parent_id,
                "hierarchy": "parent",
                "type": "table",
                "contents": pair['parent_contents'],
                "dense_embedding": parent_placeholder_dense
            })
            processed_parents[parent_id] = True
            
    # Batch embed all child contents at once
    if child_contents_to_embed:
        child_dense_embeddings = model.encode(child_contents_to_embed, show_progress_bar=True)
    else:
        child_dense_embeddings = []
    
    # Create final child chunk dictionaries
    embedding_idx = 0
    # Text children
    for pair in text_chunk_pairs:
        final_chunks.append({
            "file_id": file_id,
            "parent_id": pair['parent_id'],
            "hierarchy": "child",
            "type": "text",
            "contents": pair['child_contents'],
            "dense_embedding": child_dense_embeddings[embedding_idx].tolist()
        })
        embedding_idx += 1
        
    # Table children
    for pair in table_chunk_pairs:
        final_chunks.append({
            "file_id": file_id,
            "parent_id": pair['parent_id'],
            "hierarchy": "child",
            "type": "table",
            "contents": pair['child_contents'],
            "dense_embedding": child_dense_embeddings[embedding_idx].tolist()
        })
        embedding_idx += 1

    return final_chunks
