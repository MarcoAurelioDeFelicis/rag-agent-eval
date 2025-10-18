import os

""" Centralized static varisbles configuration  """

# --- PATHS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_FILE_PATH = os.path.join(ROOT_DIR, "data", "gz_recipe.csv")

DB_PERSIST_DIRECTORY = os.path.join(ROOT_DIR, "db")


# --- MODEL NAMES ---
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

RAG_LLM_MODEL = "gemini-2.5-pro"

JUDGE_LLM_MODEL = "gemini-2.5-pro"

SCORER_LLM_MODEL = "gemini-2.5-flash"

ANALYZER_LLM_MODEL = "gemini-2.0-flash"

# --- RAG FLOW PARAMETERS ---
DEFAULT_RERANK_TOP_N = 4
DEFAULT_RETRIEVER_TOP_K = 10

# --- FALLBACK MODEL NAMES ---
FALLBACK_LLM_MODEL = "gemini-2.5-flash"


# --- TEXT SPLITTER PARAMETERS ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- TOKENIZER PARALLELISM ---
# warning from HuggingFace tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"