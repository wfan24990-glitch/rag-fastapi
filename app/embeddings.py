import os
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from app.config import EMBEDDING_MODEL_PATH

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _resolve_model_path(mpath: str) -> str:
    """
    If mpath points to a HF hub repo cache with snapshots, choose the snapshot folder.
    Otherwise return the path or model id as-is.
    """
    if not mpath:
        return "BAAI/bge-small-zh-v1.5"
    if os.path.exists(mpath):
        snap_dir = os.path.join(mpath, "snapshots")
        if os.path.isdir(snap_dir):
            # pick first snapshot folder (usually contains the actual model files)
            entries = sorted([os.path.join(snap_dir, d) for d in os.listdir(snap_dir) if os.path.isdir(os.path.join(snap_dir, d))])
            if entries:
                return entries[0]
        # maybe the directory itself is already a snapshot-like folder
        return mpath
    # fallback to model id string
    return mpath

def load_model(force_reload: bool = False):
    global _tokenizer, _model, _device
    if _model is None or _tokenizer is None or force_reload:
        name = _resolve_model_path(EMBEDDING_MODEL_PATH)
        # If name is a path, transformers will load from local files; if it's an id, it will fetch from HF hub/cache.
        _tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        _model = AutoModel.from_pretrained(name)
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    summed_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return (summed / summed_mask)

def get_embeddings(texts: List[str], batch_size: int = 8) -> np.ndarray:
    tok, model = load_model()
    device = _device
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            out = model(**enc)
            pooled = mean_pooling(out, enc["attention_mask"])  # tensor (bs, dim)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)  # L2 normalize (useful for cosine)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)
