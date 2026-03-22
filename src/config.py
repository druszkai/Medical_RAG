import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# API keys
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
AGENTQL_API_KEY = os.environ["AGENTQL_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Paths
PY_DIR = Path(__file__).parent
AGENTS_DIR = PY_DIR / "agents"
SCRAPERS_DIR = PY_DIR / "scrapers"
TOOLS_DIR = PY_DIR / "tools"

SRC_DIR = PY_DIR.parent

DATA_DIR = SRC_DIR / "data"
DATABASE_DIR = DATA_DIR / "database"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVAL_DATA_DIR = DATA_DIR / "eval"

CHROMA_DB = DATABASE_DIR / "chroma_db"

# Embedding models
MULTILINGUAL_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MEDICAL_EMBEDDING = "abhinand/MedEmbed-small-v0.1"
OPENAI_SMALL_EMBEDDING = "text-embedding-3-small"

# LLM models
LLM_LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"
LLM_LLAMA_3 = "llama3"
LLM_LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile"
LLM_PHI_3 = "phi3"
LLM_OPENAI_GTP_OSS_20B = "openai/gpt-oss-20b"
LLM_OPENAI_GPT_OSS_120B = "openai/gpt-oss-120b"