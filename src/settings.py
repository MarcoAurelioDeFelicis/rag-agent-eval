import os

""" Centralized static varisbles configuration  """

# --- PATHS ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_FILE_PATH = os.path.join(ROOT_DIR, "data", "gz_recipe.csv")

DB_PERSIST_DIRECTORY = os.path.join(ROOT_DIR, "db")


# --- MODEL NAMES ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RAG_LLM_MODEL = "gemini-2.5-pro"

JUDGE_LLM_MODEL = "gemini-2.5-pro"

SCORER_LLM_MODEL = "gemini-2.5-flash"

# --- FALLBACK MODEL NAMES ---
FALLBACK_LLM_MODEL = "gemini-2.5-flash"


# --- TEXT SPLITTER PARAMETERS ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- TOKENIZER PARALLELISM ---
# warning from HuggingFace tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"