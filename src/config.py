import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vector_stores")

CATEGORIES = ["technical", "billing", "shipping"]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"

SEMANTIC_ROUTER_THRESHOLD = 0.65
RELEVANCE_SCORE_THRESHOLD = 0.35
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 6