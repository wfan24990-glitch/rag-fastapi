from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import time
from app.utils.chunker import chunk_text
from app.embeddings import get_embeddings
from app.vectorstore import search, add_embeddings, get_existing_sources, deduplicate_index
from app.reranker import rerank
from app.pipeline import build_rag_prompt
from app.llm import generate_local, generate_openai
from app.config import TOP_K, LLM_CONTEXT_DOCS
from app.crawler.api import router as crawler_router

router = APIRouter()
router.include_router(crawler_router, prefix="/crawler", tags=["crawler"])


class IngestRequest(BaseModel):
    text: str
    source: str = "local"
    sync: bool = False  # New field to control sync/async execution

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K

@router.get("/status")
async def status():
    return {"status": "ok"}

@router.get("/sources")
async def list_sources():
    sources = get_existing_sources()
    return {"count": len(sources), "sources": list(sources)}

@router.post("/admin/deduplicate")
async def admin_deduplicate():
    removed = deduplicate_index()
    return {"status": "completed", "removed_duplicates": removed}

def _background_ingest(chunks, source):
    embeddings = get_embeddings(chunks)
    metas = [{"source": source, "id": idx, "text": c} for idx, c in enumerate(chunks)]
    add_embeddings(embeddings, metas)
    print(f"Background ingestion completed for source: {source}")

@router.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    chunks = chunk_text(req.text, chunk_size=512, overlap=64)
    
    if req.sync:
        # Synchronous execution (blocking)
        _background_ingest(chunks, req.source)
        return {"status": "completed", "ingested_chunks_count": len(chunks), "message": "Ingestion completed synchronously"}
    else:
        # Asynchronous execution (background task)
        background_tasks.add_task(_background_ingest, chunks, req.source)
        return {"status": "processing", "ingested_chunks_count": len(chunks), "message": "Ingestion started in background"}

@router.post("/query")
async def query(req: QueryRequest):
    t0 = time.time()
    timings = {}
    
    # 1. Embedding
    t_start = time.time()
    q = req.query
    q_vec = get_embeddings([q])[0]
    timings["embedding"] = time.time() - t_start

    # 2. Vector Search
    t_start = time.time()
    candidates = search(q_vec, top_k=req.top_k)
    timings["search"] = time.time() - t_start

    if not candidates:
        return {"answer": "没有检索到相关内容。", "sources": [], "debug_info": {"timings": timings}}

    # Prepare initial candidates info for debug
    initial_candidates_info = []
    for c in candidates:
        meta = c.get("meta", {})
        initial_candidates_info.append({
            "id": meta.get("id"),
            "source": meta.get("source"),
            "text": meta.get("text", "")[:200] + "...", # Truncate for debug view
            "score": float(c.get("score", 0.0))
        })

    # 3. Rerank
    t_start = time.time()
    candidate_texts = [c["meta"].get("text","") for c in candidates]
    # rerank now returns list of dicts: {'text': str, 'score': float, 'index': int}
    reranked = rerank(q, candidate_texts, batch_size=8)
    timings["rerank"] = time.time() - t_start

    top = reranked[:LLM_CONTEXT_DOCS]
    top_for_context = []
    
    # Correctly map back to metadata using 'index'
    for item in top:
        idx = item["index"]
        score = item["score"]
        text = item["text"]
        
        # Get original metadata
        if idx < len(candidates):
            meta = candidates[idx]["meta"]
        else:
            meta = {"source": "unknown", "id": "unknown"}
            
        top_for_context.append({
            "id": meta.get("id"), 
            "source": meta.get("source"), 
            "text": text, 
            "score": score
        })

    # 4. Generation
    t_start = time.time()
    system_prompt, user_prompt = build_rag_prompt(q, top_for_context, max_snippets=LLM_CONTEXT_DOCS)
    print("DEBUG_PROMPT_SYSTEM:", system_prompt)
    print("DEBUG_PROMPT_USER:", user_prompt[:2000])
    try:
        answer = await generate_local(system_prompt, user_prompt)
    except Exception:
        answer = await generate_openai(system_prompt, user_prompt)
    timings["generation"] = time.time() - t_start
    
    timings["total"] = time.time() - t0

    # Construct response
    sources = [{"text": t["text"], "score": t["score"], "source": t["source"], "id": t["id"]} for t in top_for_context]
    
    debug_info = {
        "timings": timings,
        "retrieval": {
            "initial_candidates": initial_candidates_info,
            "reranked_candidates": [
                {
                    "id": t["id"],
                    "source": t["source"],
                    "score": t["score"],
                    "text": t["text"][:200] + "..."
                } for t in top_for_context
            ]
        }
    }

    return {"answer": answer, "sources": sources, "debug_info": debug_info}
