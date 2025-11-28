import os
import faiss
import numpy as np
import pickle

INDEX_PATH = os.path.expanduser(os.getenv("FAISS_INDEX_PATH", "~/projects/rag-fastapi/data/faiss_index.bin"))
META_PATH = INDEX_PATH + ".meta.pkl"

_index = None
_id_to_meta = {}
_dim = None

def create_index(dim):
    global _index, _dim
    _dim = dim
    _index = faiss.IndexFlatIP(dim)

def add_embeddings(embeddings, metas):
    global _index, _id_to_meta
    if _index is None:
        create_index(embeddings.shape[1])
    n_before = _index.ntotal
    _index.add(embeddings)
    for i, meta in enumerate(metas):
        _id_to_meta[n_before + i] = meta
    persist_index()

def search(query_vec, top_k=10):
    global _index
    if _index is None or _index.ntotal == 0:
        return []
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    distances, indices = _index.search(query_vec, top_k)
    results = []
    for d, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        meta = _id_to_meta.get(int(idx), {})
        results.append({"score": float(d), "id": int(idx), "meta": meta})
    return results

def persist_index():
    global _index, _id_to_meta
    if _index is None:
        return
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(_index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(_id_to_meta, f)

def load_index():
    global _index, _id_to_meta, _dim
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH, "rb") as f:
                _id_to_meta = pickle.load(f)
        _dim = _index.d if hasattr(_index, "d") else None
