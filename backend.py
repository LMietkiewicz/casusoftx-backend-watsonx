from flask import Flask, request, jsonify
from waitress import serve
from concurrent.futures import ThreadPoolExecutor
import random
import os
import fitz  # PyMuPDF
import json
import requests
import datetime
import tempfile
import zipfile
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from typing import List, Any, Dict
from utils import call_llm, search_vectors, convert_to_pdf

import config

from tasks import (
    summary, summary_formatter, category_subcategory, upload_to_milvus,
    department_assignment, orlen_department_extraction, base_extraction,
    base_extraction_formatter, check_confidential, suggested_action, other
)

encoder = SentenceTransformer("sdadas/mmlw-retrieval-roberta-large-v2")

#ollama.pull(model=MODEL)

CONTENT_TYPES = (
    "application/pdf", 
    "application/zip", 
    "application/x-zip-compressed",
    "application/doc", 
    "application/msword",
    "application/docx", 
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/txt", 
    "text/plain",
    "application/xml", 
    "text/xml"
)

app = Flask(__name__)
executor = ThreadPoolExecutor()

def process_zip_file(zip_path: str) -> Tuple[str, bytes]:
    """
    Process a zip file containing PDFs, XML, and TXT files.
    
    Args:
        zip_path (str): Path to the zip file
        
    Returns:
        Tuple[str, bytes]: A tuple containing:
            - Combined string of all file contents with labels
            - Merged PDF as bytes
    """
    try:
        combined_text = ""
        pdf_files = []
        #xml_content = ""
        #txt_content = ""
        
        # Extract and process files from zip
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            file_list = zip_file.namelist()
            
            # Sort files to ensure consistent ordering
            pdf_files_info = []
            #xml_files_info = []
            #txt_files_info = []
            
            for filename in sorted(file_list):
                if filename.lower().endswith('.pdf'):
                    pdf_files_info.append(filename)
                '''elif filename.lower().endswith('.xml'):
                    xml_files_info.append(filename)
                elif filename.lower().endswith('.txt'):
                    txt_files_info.append(filename)'''
            
            # Process PDF files
            for i, pdf_filename in enumerate(pdf_files_info, 1):
                with zip_file.open(pdf_filename) as pdf_file:
                    pdf_content = pdf_file.read()
                    pdf_files.append((pdf_filename, pdf_content))
                    
                    # Extract text from PDF for the combined string
                    try:
                        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
                        pdf_text = ""
                        for page_num in range(pdf_doc.page_count):
                            page = pdf_doc[page_num]
                            pdf_text += page.get_text() + "\n"
                        pdf_doc.close()
                        
                        combined_text += pdf_text + "\n\n"
                    except Exception as e:
                        combined_text += f"PDF_{i} ({pdf_filename}):\n[Error reading PDF: {str(e)}]\n\n"
            
            # Process XML files
            '''for i, xml_filename in enumerate(xml_files_info, 1):
                with zip_file.open(xml_filename) as xml_file:
                    try:
                        xml_content = xml_file.read().decode('utf-8')
                        if len(xml_files_info) > 1:
                            combined_text += f"XML_{i} ({xml_filename}):\n{xml_content}\n\n"
                        else:
                            combined_text += f"XML ({xml_filename}):\n{xml_content}\n\n"
                    except Exception as e:
                        combined_text += f"XML ({xml_filename}):\n[Error reading XML: {str(e)}]\n\n"'''
            
            # Process TXT files
            '''for i, txt_filename in enumerate(txt_files_info, 1):
                with zip_file.open(txt_filename) as txt_file:
                    try:
                        txt_content = txt_file.read().decode('utf-8')
                        if len(txt_files_info) > 1:
                            combined_text += f"TXT_{i} ({txt_filename}):\n{txt_content}\n\n"
                        else:
                            combined_text += f"TXT ({txt_filename}):\n{txt_content}\n\n"
                    except Exception as e:
                        combined_text += f"TXT ({txt_filename}):\n[Error reading TXT: {str(e)}]\n\n"'''
        
        # Create merged PDF
        merged_pdf_bytes = create_merged_pdf(pdf_files)
    except Exception as e:
        print(f"Error processing zip file: {str(e)}")
    
    return merged_pdf_bytes

def create_merged_pdf(pdf_files: List[Tuple[str, bytes]]) -> bytes:
    """
    Merge multiple PDFs into a single PDF with minimal formatting.
    
    Args:
        pdf_files: List of tuples (filename, pdf_content_bytes)
        
    Returns:
        bytes: Merged PDF as bytes
    """
    if not pdf_files: #this ought to be changed at some point
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "No PDF files found", fontsize=12)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    
    merged_doc = fitz.open()
    
    for filename, pdf_content in pdf_files:
        try:
            # Add simple text separator page
            sep_page = merged_doc.new_page()
            sep_page.insert_text((50, 100), f"------", fontsize=12)
            
            # Merge PDF content
            source_doc = fitz.open(stream=pdf_content, filetype="pdf")
            merged_doc.insert_pdf(source_doc)
            source_doc.close()
            
        except Exception as e:
            # Add minimal error page
            error_page = merged_doc.new_page()
            error_page.insert_text((50, 100), f"Error: {filename}", fontsize=12)
    
    pdf_bytes = merged_doc.tobytes()
    merged_doc.close()
    return pdf_bytes

def generate_task_id():
    """Generate a consistent task ID based on current datetime with constant length of 14 characters."""
    time = datetime.datetime.now()
    return time.strftime("%y%m%d%H%M%S%f")[10:]

def send_callback(url, payload):
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Callback to {url} failed: {e}")


def run_rag_with_context(query: str, context_fragments: List[Dict[str, Any]], is_global_search: bool) -> str:
    """
    Constructs a prompt with context and sends it to the LLM for a response.

    Args:
        query (str): The user's query.
        context_fragments (List[Dict[str, Any]]): The list of parent chunks retrieved from Milvus.
        is_global_search (bool): True if the search was across all documents, False if it was document-specific.

    Returns:
        str: The natural language response from the LLM.
    """
    try:
        # --- Prompt Engineering ---
        
        # 1. Format the context fragments for the prompt
        formatted_context = ""
        for i, fragment in enumerate(context_fragments):
            file_id = fragment.get("file_id", "N/A")
            contents = fragment.get("contents", "")
            
            # For global search, we include the file_id in the context.
            if is_global_search:
                formatted_context += f"--- Fragment z dokumentu o ID: {file_id} ---\n"
            else:
                 formatted_context += f"--- Fragment {i+1} ---\n"
            
            formatted_context += f"{contents}\n\n"

        # 2. Construct the system prompt with the new strategy
        # Start with the English "magic phrase" for the qwen model
        if "qwen" in config.MODEL.lower():
            system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
            system_prompt += "Od teraz, będziesz komunikować się wyłącznie w języku polskim. \n"
        else:
            system_prompt = ""

        system_prompt += "Jesteś inteligentnym asystentem specjalizującym się w analizie dokumentów. Twoim zadaniem jest udzielanie precyzyjnych i pomocnych odpowiedzi na podstawie dostarczonych fragmentów.\n\n"
        system_prompt += "KONTEKST:\n"
        system_prompt += "Poniżej znajdują się fragmenty dokumentów, które posłużą jako podstawa Twojej odpowiedzi:\n"
        system_prompt += formatted_context
        
        system_prompt += "INSTRUKCJE:\n"
        system_prompt += "1. Odpowiedz na pytanie użytkownika w naturalny, uprzejmy i profesjonalny sposób.\n"
        system_prompt += "2. Twoja odpowiedź musi być zwięzła i oparta wyłącznie na informacjach zawartych w powyższym KONTEKście.\n"
        
        # Add the sourcing instruction only for global searches
        if is_global_search:
            system_prompt += "3. Na końcu swojej odpowiedzi, wskaż identyfikatory (ID) dokumentów, z których pochodzą informacje, np. 'Informacje pochodzą z dokumentów o ID: doc_001, doc_005.'\n"
 
         # 3. Set model options
        options = {
            "temperature": 0.2,
            "top_k": 20,
            "top_p": 0.5,
            "num_predict": 1024,
            "repeat_penalty": 1.1
        }

        # Call the LLM
        llm_answer = call_llm(
            endpoint="generate",
            input=query,
            system_message=system_prompt,
            options=options
        )

        return llm_answer

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request to Ollama failed: {e}")
        return "Przepraszam, wystąpił błąd podczas komunikacji z modelem językowym."
    except Exception as e:
        print(f"[ERROR] An error occurred in run_rag_with_context: {e}")
        return "Przepraszam, wystąpił nieoczekiwany błąd."


def handle_task(document_id, task_type, task_id, params, success_cb, error_cb, precomputed_summary):
    try:
        task_result = {}

        if task_type == "SUMMARY":
            summary_result = precomputed_summary
            summary_result = summary_formatter(summary_result)
            task_result = {"results": {"SUMMARY": summary_result}}
            print(task_result)

        elif task_type == "CATEGORY_SUBCATEGORY":
            summary_result = precomputed_summary
            category_data = params.get("CATEGORY_SUBCATEGORY", {}).get("categories", [])
            formatted_categories = {cat["name"]: [s["name"] for s in cat.get("subcategories", [])] for cat in category_data}
            found_category, found_subcategory = category_subcategory(summary_result, formatted_categories)
            results_dict = {"CATEGORY": found_category}
            if found_subcategory:
                results_dict["SUBCATEGORY"] = found_subcategory
            task_result = {"results": {"CATEGORY_SUBCATEGORY": results_dict}}
            print(task_result)

        elif task_type == "DEPARTMENT_ASSIGNMENT":
            summary_result = precomputed_summary  
            departments_data = params.get("DEPARTMENT_ASSIGNMENT", {}).get("departments", [])
            print(departments_data)
            formatted_departments = {"departments": departments_data}
            task_result = {"results": {"DEPARTMENT_ASSIGNMENT": department_assignment(summary_result, formatted_departments)}}
            print(task_result) 

        elif task_type == "ORLEN_DEPARTMENT_EXTRACTION":
            summary_result = precomputed_summary
            task_result = {"results": {"ORLEN_DEPARTMENT_EXTRACTION": orlen_department_extraction(summary_result)}}
            print(task_result)

        elif task_type == "BASE_EXTRACTION":
            summary_result = precomputed_summary
            response = base_extraction(summary_result)
            response = base_extraction_formatter(response)
            task_result = {"results": {"BASE_EXTRACTION": response}}
            print(task_result)

        elif task_type == "CHECK_CONFIDENTIAL":
            summary_result = precomputed_summary
            task_result = {"results": {"CHECK_CONFIDENTIAL": check_confidential(summary_result)}}
            print(task_result)

        elif task_type == "OTHER":
            prompt = params.get("OTHER", {}).get("query", "")
            task_result = {"results": {"OTHER": other(prompt)}}
            print(task_result)

        elif task_type == "SUGGESTED_ACTION":
            summary_result = precomputed_summary
            task_result = {"results": {"SUGGESTED_ACTION": suggested_action(summary_result)}}
            print(task_result)

        else:
            print(f"Unsupported task type: {task_type}")
            raise ValueError("Unsupported task type")

        payload = {
            "documentId": document_id,
            "taskId": task_id,
            "taskType": task_type,
            **task_result
        }

        print(payload)
        send_callback(success_cb, payload)

    except Exception as e:
        payload = {
            "documentId": document_id,
            "taskId": task_id,
            "taskType": task_type,
            "error": str(e)
        }

        print(payload)
        send_callback(error_cb, payload)

@app.route("/api/task", methods=["POST"])
def create_task():
    try:
        document_id = int(request.form["documentId"])
        content_type = request.form["documentContentType"]
        tasks = json.loads(request.form["tasks"])
        params = json.loads(request.form.get("params", "{}"))
        success_cb = request.form["successCallbackUrl"]
        error_cb = request.form["errorCallbackUrl"]

        if content_type not in CONTENT_TYPES:
            return jsonify({"error": f"Unsupported Media Type: {content_type}"}), 415

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{content_type.split("/")[-1]}')
        file = request.files["documentFile"]
        file.save(temp_file.name)
        temp_file.close()
        temp_file_path = temp_file.name

        # Generate taskIds up front and return them immediately
        task_dicts = []
        i = 0
        for task_type in tasks:
            task_id = generate_task_id() + str(i)
            task_dicts.append({"taskType": task_type, "taskId": task_id})
            i += 1

        def background_processing():
            try:
                if content_type == "application/x-zip-compressed" or content_type == "application/zip":
                    # Process the ZIP file directly, don't use fitz.open()
                    merged_pdf_bytes = process_zip_file(temp_file_path)
                    
                    # Create a PDF document from the merged bytes
                    with fitz.open(stream=merged_pdf_bytes, filetype="pdf") as doc: 
                        """for page_num in range(doc.page_count):
                            page = doc[page_num]
                            text = page.get_text()
                            print(f"--- Page {page_num + 1} ---")
                            print(text)
                            print("\n" + "-" * 80 + "\n")"""
                        upload_to_milvus(doc, document_id, encoder)

                    with fitz.open(stream=merged_pdf_bytes, filetype="pdf") as doc:
                        precomputed_summary = summary(doc)
                    
                    for task_dict in task_dicts:
                        executor.submit(
                            handle_task,
                            document_id,
                            task_dict["taskType"],
                            task_dict["taskId"],
                            params,
                            success_cb,
                            error_cb,
                            precomputed_summary
                        )

                elif content_type == "application/pdf":
                    with fitz.open(temp_file_path) as doc:
                        upload_to_milvus(doc, document_id, encoder)

                    with fitz.open(temp_file_path) as doc:    
                        precomputed_summary = summary(doc)
                    
                    for task_dict in task_dicts:
                        executor.submit(
                            handle_task,
                            document_id,
                            task_dict["taskType"],
                            task_dict["taskId"],
                            params,
                            success_cb,
                            error_cb,
                            precomputed_summary
                        )

                elif content_type in ("application/doc", "application/msword",
                                    "application/docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    "application/txt", "text/plain", 
                                    "application/xml", "text/xml"):
                    # Convert to PDF first
                    pdf_bytes = convert_to_pdf(temp_file_path, content_type)
                    
                    # Process as PDF
                    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                        upload_to_milvus(doc, document_id, encoder)

                    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:    
                        precomputed_summary = summary(doc)
                    
                    for task_dict in task_dicts:
                        executor.submit(
                            handle_task,
                            document_id,
                            task_dict["taskType"],
                            task_dict["taskId"],
                            params,
                            success_cb,
                            error_cb,
                            precomputed_summary
                        )

                else:
                    raise ValueError(f"Unsupported file type for processing: {content_type}")

            except Exception as e:
                print(f"[ERROR] Background processing failed: {e}")
                send_callback(error_cb, {
                    "documentId": document_id,
                    "taskType": "error",
                    "error": str(e)
                })

            finally:
                print(f"Cleaning up temporary file: {temp_file_path}")
                os.remove(temp_file_path)

        executor.submit(background_processing)

        # Return immediately
        return jsonify(task_dicts), 200

    except KeyError as ke:
        return jsonify({"error": f"Missing required field: {ke}"}), 400
    except json.JSONDecodeError as je:
        return jsonify({"error": "Invalid JSON in 'tasks' or 'params' field."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def sync_query():
    try:
        data = request.get_json()
        document_id = data.get("documentId")
        document_id = int(document_id) if document_id is not None else None
        task_type = "OTHER"
        task_id = generate_task_id() + str(random.randint(0, 10))
        query = data["query"]
        
        try:
            context_fragments = search_vectors(query, encoder, document_id)
            if not context_fragments:
                return jsonify({"error": "No context found"}), 404
            
            # Determine if it was a global search
            is_global = document_id is None

            response = run_rag_with_context(query, context_fragments, is_global)

            payload = {
                "task_type": task_type,
                "task_id": task_id,
                "response": response,
                "sources": context_fragments
            }

            return jsonify(payload), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Use the port defined in your config file
    port = config.APP_PORT
    try:
        print(f"Starting backend server on host 0.0.0.0, port {port}...")
        serve(app, host="0.0.0.0", port=port)

    except Exception as e:
        # This will catch errors during startup.
        print(f"An error occurred during server startup or execution: {e}")