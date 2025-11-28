from dotenv import load_dotenv
import os

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

DEFAULT_HF_CACHE_BGE_SMALL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"
)

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", DEFAULT_HF_CACHE_BGE_SMALL)

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.expanduser("~/projects/rag-fastapi/data/faiss_index.bin"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TOP_K = int(os.getenv("TOP_K", "20"))
LLM_CONTEXT_DOCS = int(os.getenv("LLM_CONTEXT_DOCS", "5"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
