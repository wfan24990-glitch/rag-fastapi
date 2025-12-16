import os
import faiss
import numpy as np
import pickle
import threading

INDEX_PATH = os.path.expanduser(os.getenv("FAISS_INDEX_PATH", "~/projects/rag-fastapi/data/faiss_index.bin"))
META_PATH = INDEX_PATH + ".meta.pkl"

_index = None
_id_to_meta = {}
_dim = None
_lock = threading.Lock()

def create_index(dim):
    global _index, _dim
    _dim = dim
    _index = faiss.IndexFlatIP(dim)

def add_embeddings(embeddings, metas):
    global _index, _id_to_meta
    with _lock:
        if _index is None:
            create_index(embeddings.shape[1])
        n_before = _index.ntotal
        _index.add(embeddings)
        for i, meta in enumerate(metas):
            _id_to_meta[n_before + i] = meta
        persist_index()

def search(query_vec, top_k=10):
    global _index
    if _index is None:
        load_index()
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
    # make a shallow copy to avoid mutation during pickle
    meta_copy = dict(_id_to_meta)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta_copy, f)

def load_index():
    global _index, _id_to_meta, _dim
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH, "rb") as f:
                _id_to_meta = pickle.load(f)
        _dim = _index.d if hasattr(_index, "d") else None

def get_existing_sources():
    """Return a set of all sources currently in the index."""
    global _id_to_meta
    if _index is None:
        load_index()
    sources = set()
    for meta in _id_to_meta.values():
        if "source" in meta:
            sources.add(meta["source"])
    return sources

def deduplicate_index():
    """
    Remove duplicate entries based on 'source' and 'text'.
    Rebuilds the index in memory and persists it.
    """
    global _index, _id_to_meta
    with _lock:
        if _index is None:
            load_index()
        if _index is None or _index.ntotal == 0:
            return 0

        print("Starting deduplication...")
        # Group by source
        source_groups = {}
        for idx, m in _id_to_meta.items():
            src = m.get('source', 'unknown')
            
            # Normalize source: if it's just a number like "315", convert to "315.json"
            # This handles the case where "315" and "315.json" are duplicates
            if src.isdigit():
                src = f"{src}.json"
                # Update the meta in place so the kept record has the correct source name
                m['source'] = src
                
            if src not in source_groups:
                source_groups[src] = []
            source_groups[src].append(idx)

        kept_vectors = []
        kept_metas = []
        removed_count = 0

        # Reconstruct all vectors (fast for IndexFlat)
        try:
            all_vectors = _index.reconstruct_n(0, _index.ntotal)
        except Exception as e:
            print(f"Error reconstructing vectors: {e}")
            return 0

        for src, indices in source_groups.items():
            seen_texts = set()
            for idx in indices:
                txt = _id_to_meta[idx].get('text', '')
                # Deduplicate by exact text match within the same source
                if txt not in seen_texts:
                    seen_texts.add(txt)
                    kept_vectors.append(all_vectors[idx])
                    kept_metas.append(_id_to_meta[idx])
                else:
                    removed_count += 1

        if removed_count > 0:
            print(f"Removing {removed_count} duplicates. Rebuilding index...")
            dim = _index.d
            new_index = faiss.IndexFlatIP(dim)
            if kept_vectors:
                new_index.add(np.array(kept_vectors))
            
            _index = new_index
            _id_to_meta = {i: m for i, m in enumerate(kept_metas)}
            persist_index()
            print("Deduplication complete.")
        else:
            print("No duplicates found.")
            
        return removed_count
