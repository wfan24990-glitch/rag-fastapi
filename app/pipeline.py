from typing import List, Dict, Tuple

def build_rag_prompt(question: str, snippets: List[Dict], max_snippets: int = 3) -> Tuple[str, str]:
    """
    Build a robust RAG prompt:
    - question: user question
    - snippets: list of dicts each must contain keys: id, source, text, score (score optional)
    - max_snippets: number of top snippets to include (control token usage)
    Returns: (system_prompt, user_prompt)
    """
    system = (
        "You are an expert AI assistant specialized in Question Answering over specific documents. "
        "Your task is to answer the user's question based ONLY on the provided context snippets.\n\n"
        "Guidelines:\n"
        "1. Use ONLY the provided context. Do not use external knowledge or prior training data.\n"
        "2. If the answer cannot be derived from the context, strictly say: 'I cannot answer this question based on the provided documents.'\n"
        "3. Cite your sources. Every factual statement must be backed by a citation in the format [source:source_name#id].\n"
        "4. Keep the answer concise, professional, and directly relevant to the question."
    )

    # Sort snippets by score if provided, else keep order
    if snippets and 'score' in snippets[0]:
        snippets_sorted = sorted(snippets, key=lambda s: s.get('score', 0), reverse=True)
    else:
        snippets_sorted = snippets

    selected = snippets_sorted[:max_snippets]

    context_lines = []
    for i, s in enumerate(selected, start=1):
        sid = s.get('id', f"chunk{i}")
        src = s.get('source', 'unknown')
        text = s.get('text', '').strip()
        # Truncate snippet text to reasonable length per snippet to limit tokens
        max_chars = 1200
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        context_lines.append(f"<snippet id=\"{sid}\" source=\"{src}\">\n{text}\n</snippet>")

    context_block = "\n\n".join(context_lines)

    user = (
        f"User Question: {question}\n\n"
        f"Context Snippets:\n{context_block}\n\n"
        "Please answer the question following the guidelines above."
    )

    return system, user
