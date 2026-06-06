"""config.py - central configuration management for environment variables, API keys, and application settings"""
import os
from dotenv import load_dotenv

# --- Loading Dotenv ---
load_dotenv()

# --- Provider Configuration ---
API_KEY = os.getenv("API_KEY", "")  
PROJECT_ID = os.getenv("PROJECT_ID", "")

INFERENCE_PROVIDER_BASE_URL = os.getenv("INFERENCE_PROVIDER_BASE_URL")
INFERENCE_PROVIDER_SSL_VERIFY = os.getenv("INFERENCE_PROVIDER_SSL_VERIFY", "true").lower() == "true"

# --- Milvus Configuration ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))
CONNECTION_ALIAS = "default"

# --- Application Port ---
APP_PORT = int(os.getenv("PORT", 5000))

# --- Model Configuration ---
MODEL = os.getenv("MODEL", "")