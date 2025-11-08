import requests
from ibm_cloud_sdk_core import IAMTokenManager
import config
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class ModelInferanceChat:
    def __init__(self):
        self.access_token = IAMTokenManager(apikey=config.API_KEY).get_token()
        self.project_id = config.PROJECT_ID

    def chat(
        self, 
        model_id, 
        messages, 
        **kwargs
    ):
        wml_url = f"{config.INFERENCE_PROVIDER_BASE_URL}/ml/v1/text/chat?version=2024-10-07"
        Headers = {
            "Authorization": "Bearer " + self.access_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "model_id": model_id,
            "messages": messages,
            "project_id": self.project_id,
        }
        data = data | kwargs
        response = requests.post(wml_url, json=data, headers=Headers)
        if response.status_code == 200:
            return response.json()
        else:
            return response.text
        
def call_llm(
    endpoint: str,
    input: str,
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
            "temperature": 0.2,
            "top_k": 20,
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
            chat_client = ModelInferanceChat()
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input}
            ]
            
            response = chat_client.chat(
                model_id=config.MODEL,
                messages=messages,
                temperature=options["temperature"],
                top_k=options["top_k"],
                top_p=options["top_p"],
                max_tokens=options["num_predict"],
                repetition_penalty=options["repeat_penalty"],
                stream=False
            )
            
            return response.get("generated_text", "").strip()
            
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