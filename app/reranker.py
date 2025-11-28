import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
from app.config import RERANKER_MODEL

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None

def load_reranker():
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
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

def rerank(query: str, candidates: List[str], batch_size: int = 8) -> List[Tuple[str, float]]:
    tokenizer, model = load_reranker()
    pairs_scores = []
    # prepare pair lists
    for i in range(0, len(candidates), batch_size):
        batch_docs = candidates[i:i+batch_size]
        queries = [query] * len(batch_docs)
        try:
            scores = _score_batch(tokenizer, model, queries, batch_docs)
        except Exception:
            # fallback: return zeros for this batch if model fails
            scores = [0.0] * len(batch_docs)
        pairs_scores.extend(list(zip(batch_docs, scores)))
    # sort by score desc
    pairs_scores_sorted = sorted(pairs_scores, key=lambda x: x[1], reverse=True)
    return pairs_scores_sorted
