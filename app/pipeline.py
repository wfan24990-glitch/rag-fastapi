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
        "你是一名面向特定文档的中文问答助手，只能依据给定的上下文片段回答用户问题。\n\n"
        "回答规则：\n"
        "1) 严格使用提供的上下文，不得调用外部知识或模型自带知识。\n"
        "2) 若上下文无法回答，必须回复：\"我无法基于提供的文档回答这个问题。\"，不要编造。\n"
        "3) 需给出引用，格式为 [source:source_name#id]，每条事实都要对应引用。\n"
        "4) 用中文作答，保持简洁、专业、紧扣问题。"
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
