from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.utils.chunker import chunk_text
from app.embeddings import get_embeddings
from app.vectorstore import search, add_embeddings
from app.reranker import rerank
from app.pipeline import build_rag_prompt
from app.llm import generate_local, generate_openai
from app.config import TOP_K, LLM_CONTEXT_DOCS

router = APIRouter()

class IngestRequest(BaseModel):
    text: str
    source: str = "local"

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K

@router.get("/status")
async def status():
    return {"status": "ok"}

def _background_ingest(chunks, source):
    embeddings = get_embeddings(chunks)
    metas = [{"source": source, "id": idx, "text": c} for idx, c in enumerate(chunks)]
    add_embeddings(embeddings, metas)
    print(f"Background ingestion completed for source: {source}")

@router.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    chunks = chunk_text(req.text, chunk_size=512, overlap=64)
    background_tasks.add_task(_background_ingest, chunks, req.source)
    return {"status": "processing", "ingested_chunks_count": len(chunks), "message": "Ingestion started in background"}

@router.post("/query")
async def query(req: QueryRequest):
    q = req.query
    q_vec = get_embeddings([q])[0]
    candidates = search(q_vec, top_k=req.top_k)
    if not candidates:
        return {"answer": "没有检索到相关内容。", "sources": []}
    candidate_texts = [c["meta"].get("text","") for c in candidates]
    reranked = rerank(q, candidate_texts, batch_size=8)
    top = reranked[:LLM_CONTEXT_DOCS]
    top_for_context = []
    for i, (text, score) in enumerate(top):
        meta = candidates[i]["meta"] if i < len(candidates) else {"source":"unknown","id":i}
        top_for_context.append({"id": meta.get("id", i), "source": meta.get("source","unknown"), "text": text, "score": score})
    system_prompt, user_prompt = build_rag_prompt(q, top_for_context, max_snippets=LLM_CONTEXT_DOCS)
    print("DEBUG_PROMPT_SYSTEM:", system_prompt)
    print("DEBUG_PROMPT_USER:", user_prompt[:2000])
    try:
        answer = await generate_local(system_prompt, user_prompt)
    except Exception:
        answer = await generate_openai(system_prompt, user_prompt)
    sources = [{"text": t, "score": s, "source": top_for_context[idx]["source"], "id": top_for_context[idx]["id"]} for idx,(t,s) in enumerate(top)]
    return {"answer": answer, "sources": sources}
