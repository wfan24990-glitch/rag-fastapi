import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict
from app.config import RERANKER_MODEL

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None

def _resolve_model_path(mpath: str) -> str:
    """
    If mpath points to a HF hub repo cache with snapshots, choose the snapshot folder.
    Otherwise return the path or model id as-is.
    """
    if not mpath:
        return "BAAI/bge-reranker-base"
    if os.path.exists(mpath):
        snap_dir = os.path.join(mpath, "snapshots")
        if os.path.isdir(snap_dir):
            entries = sorted(
                [
                    os.path.join(snap_dir, d)
                    for d in os.listdir(snap_dir)
                    if os.path.isdir(os.path.join(snap_dir, d))
                ]
            )
            if entries:
                return entries[0]
        return mpath
    return mpath

def load_reranker():
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        name = _resolve_model_path(RERANKER_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(name)
        _model = AutoModelForSequenceClassification.from_pretrained(name)
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model

def _score_batch(tokenizer, model, queries: List[str], docs: List[str]) -> List[float]:
    enc = tokenizer(queries, docs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    enc = {k: v.to(_device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits  # shape (batch, num_labels) or (batch,1)
        if logits is None:
            return [0.0] * len(queries)
        if logits.dim() == 1:
            scores = logits.cpu().tolist()
        elif logits.size(1) == 1:
            scores = logits.squeeze(-1).cpu().tolist()
        else:
            probs = torch.nn.functional.softmax(logits, dim=1)
            scores = probs[:, -1].cpu().tolist()
    return scores

def rerank(query: str, candidates: List[str], batch_size: int = 8) -> List[Dict]:
    """
    Rerank candidates based on query.
    Returns a list of dicts: [{'text': str, 'score': float, 'index': int}, ...]
    sorted by score descending.
    """
    tokenizer, model = load_reranker()
    results = []
    
    for i in range(0, len(candidates), batch_size):
        batch_docs = candidates[i:i+batch_size]
        queries = [query] * len(batch_docs)
        try:
            scores = _score_batch(tokenizer, model, queries, batch_docs)
        except Exception:
            # fallback: return zeros for this batch if model fails
            scores = [0.0] * len(batch_docs)
        
        for j, score in enumerate(scores):
            results.append({
                "text": batch_docs[j],
                "score": score,
                "index": i + j
            })

    # sort by score desc
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    return results_sorted
