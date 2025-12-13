from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai import Credentials
import requests
import config
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from docx import Document
import fitz
import subprocess
        
def call_llm(
    input: str,
    endpoint: str = "generate",
    system_message: str = "",
    options: Dict[str, Any] = None
) -> str:
    """
    Calls the LLM using either Ollama or Watsonx based on configuration.
    
    Args:
        user_message: The user's prompt/query
        system_message: System prompt for context/instructions
        options: Dictionary with model parameters (temperature, top_k, top_p, num_predict, repeat_penalty)
        
    Returns:
        str: The generated response from the LLM
        
    Raises:
        requests.exceptions.RequestException: If API call fails
        Exception: For other unexpected errors
    """
    if options is None:
        options = {
            "temperature": 0.5,
            "top_p": 0.5,
            "num_predict": 1024,
            "repeat_penalty": 1.1
        }
    
    try:
        if config.API_KEY == "":
            # Using Ollama
            payload = {
                "model": config.MODEL,
                "prompt": input, 
                "options": options,
                "system": system_message,
                "stream": False
            }
             
            response = requests.post(
                f"{config.INFERENCE_PROVIDER_BASE_URL}/api/{endpoint}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
                verify=False
            )
            
            response.raise_for_status()
            json_response = response.json()

            return json_response.get("response", "").strip()
            
        else:
            # Using IBM Watsonx
            deployment_inference = ModelInference(
                model_id=config.MODEL,
                credentials=Credentials(
                    api_key=config.API_KEY, url=config.INFERENCE_PROVIDER_BASE_URL
                ),
                project_id=config.PROJECT_ID,
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input}
            ]

            params = TextChatParameters(
                temperature=options["temperature"],
                max_tokens=options["num_predict"],
                top_p=options["top_p"],
                repetition_penalty=options["repeat_penalty"]
            )
            
            response = deployment_inference.chat(
                messages=messages,
                params=params
            )
            
            return response["choices"][0]["message"]["content"].strip()
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in call_llm: {e}")
        raise

def search_vectors(query_text: str, encoder: SentenceTransformer, document_id: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
    
    collection_name = "file_embeddings"
    
    try:
        print("Connecting to Milvus...")
        
        # Use MilvusClient for hybrid search
        client = MilvusClient(
            uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"
        )
        
        print("Successfully connected to Milvus.")

        # 1. --- Prepare Hybrid Search: Dense + Sparse (BM25) ---
        
        # Generate dense embedding for semantic search
        query_embedding = encoder.encode(query_text).tolist()

        # Construct the filter expression for child chunks only
        filter_expr = "hierarchy == 'child'"
        if document_id:
            filter_expr += f" and file_id == '{document_id}'"  # String fields need quotes

        # Create Dense Search Request (Semantic Search)
        dense_search_params = {
            "metric_type": "IP"
        }

        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="dense_embedding",
            param=dense_search_params,
            limit=top_k * 2,  # Retrieve more candidates for better fusion
            expr=filter_expr
        )

        # Create Sparse Search Request (BM25 Full-Text Search)
        sparse_search_params = {
            "metric_type": "BM25"
        }

        sparse_req = AnnSearchRequest(
            data=[query_text],  # Milvus converts text to BM25 sparse vector automatically
            anns_field="sparse_embedding",
            param=sparse_search_params,
            limit=top_k * 2,  # Retrieve more candidates for better fusion
            expr=filter_expr
        )

        # Perform Hybrid Search with Reciprocal Rank Fusion
        print(f"Performing hybrid search for: '{query_text}'")
        child_search_results = client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],  # Combine both search strategies
            ranker=RRFRanker(),  # Reciprocal Rank Fusion for combining results
            limit=top_k,  # Final number of results after fusion
            output_fields=["parent_id"],
            consistency_level="Strong"
        )

        # 2. --- Retrieve the corresponding PARENT chunks ---
        
        # Collect the unique parent IDs from the hybrid search results
        parent_ids = set()
        for hits in child_search_results:
            for hit in hits:
                print(f"Child hit - Score: {hit.get('distance', 'N/A')}, Parent ID: {hit.get('entity', {}).get('parent_id')}")
                parent_id = hit.get("entity", {}).get("parent_id")
                if parent_id is not None:
                    parent_ids.add(parent_id)
        
        if not parent_ids:
            print("No relevant child chunks found.")
            return []

        print(f"Found {len(parent_ids)} unique parent IDs to retrieve")

        # Query for parent chunks using collected IDs
        parent_expr = f"parent_id in {list(parent_ids)} and hierarchy == 'parent'"
        
        parent_results = client.query(
            collection_name=collection_name,
            filter=parent_expr,
            output_fields=["parent_id", "contents", "file_id", "type"],
            consistency_level="Strong"
        )
        
        print(f"Retrieved {len(parent_results)} parent chunks")
        return parent_results

    except Exception as e:
        print(f"[ERROR] Milvus hybrid search failed: {e}")
        import traceback
        traceback.print_exc()
        return []
    
def convert_to_pdf(file_path: str, content_type: str) -> bytes:
    """
    Convert various file formats to PDF.
    
    Args:
        file_path (str): Path to the input file
        content_type (str): MIME type of the input file
        
    Returns:
        bytes: PDF file as bytes
        
    Raises:
        ValueError: If the content type is not supported
        RuntimeError: If conversion fails
    """
    try:
        # If already PDF, just return the bytes
        if content_type == "application/pdf":
            with open(file_path, 'rb') as f:
                return f.read()
        
        # Handle text files (TXT, XML)
        elif content_type in ("application/txt", "text/plain", "application/xml", "text/xml"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Create PDF from text
            doc = fitz.open()
            page = doc.new_page(width=595, height=842)  # A4 size
            rect = fitz.Rect(50, 50, 545, 792)  # Margins
            page.insert_textbox(rect, text_content, fontsize=10, fontname="helv")
            
            pdf_bytes = doc.tobytes()
            doc.close()
            return pdf_bytes
        
        # Handle DOCX files
        elif content_type in ("application/docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            doc_reader = Document(file_path)
            
            # Extract text from DOCX
            full_text = []
            for para in doc_reader.paragraphs:
                full_text.append(para.text)
            text_content = '\n'.join(full_text)
            
            # Create PDF
            pdf_doc = fitz.open()
            page = pdf_doc.new_page(width=595, height=842)
            rect = fitz.Rect(50, 50, 545, 792)
            page.insert_textbox(rect, text_content, fontsize=10, fontname="helv")
            
            pdf_bytes = pdf_doc.tobytes()
            pdf_doc.close()
            return pdf_bytes
        
        # Handle DOC files (requires antiword)
        elif content_type in ("application/doc", "application/msword"):
            try:
                # Use antiword for conversion
                result = subprocess.run(
                    ['antiword', file_path],
                    check=True,
                    capture_output=True,
                    timeout=30
                )
                text_content = result.stdout.decode('utf-8', errors='ignore')
                
                # Create PDF from extracted text
                pdf_doc = fitz.open()
                page = pdf_doc.new_page(width=595, height=842)
                rect = fitz.Rect(50, 50, 545, 792)
                page.insert_textbox(rect, text_content, fontsize=10, fontname="helv")
                
                pdf_bytes = pdf_doc.tobytes()
                pdf_doc.close()
                return pdf_bytes
                
            except FileNotFoundError:
                raise RuntimeError("antiword not found. Install with: apt-get install antiword")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"antiword conversion failed: {e.stderr.decode()}")
        
        else:
            raise ValueError(f"Unsupported content type for conversion: {content_type}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to convert file to PDF: {str(e)}")