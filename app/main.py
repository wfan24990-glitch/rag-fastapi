import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import router as api_router
from app.embeddings import load_model
from app.reranker import load_reranker
from app.vectorstore import load_index

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading FAISS index...")
    load_index()
    print("Loading embedding model...")
    load_model()
    print("Loading reranker model...")
    load_reranker()
    yield

app = FastAPI(title="RAG FastAPI", lifespan=lifespan)
app.include_router(api_router, prefix="")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
