# Copilot Instructions for AI Coding Agents

## 项目架构概览
- 本项目为基于 FastAPI 的 RAG（Retrieval-Augmented Generation）服务，核心功能包括文本分块、嵌入生成、向量检索、重排序、提示构建和大模型问答。
- 主要目录结构：
  - `app/`：所有后端核心逻辑，包括 API 路由、嵌入、向量存储、重排序、RAG pipeline、LLM 接口等。
  - `app/utils/`：工具类与通用方法（如文本分块）。
  - `data/`、`models/`、`docker/`：分别用于数据、模型和容器相关内容。

## 关键组件与数据流
- API 入口：`app/api.py`，定义 `/ingest`（文本入库）和 `/query`（检索问答）等路由。
- 数据流：
  1. `/ingest`：文本分块 → 生成嵌入 → 存入向量库（`add_embeddings`）。
  2. `/query`：问题嵌入 → 向量检索 → 重排序 → 构建 RAG Prompt → LLM 生成答案。
- 各模块间通过函数调用和数据结构（如嵌入、meta 信息）进行通信。

## 开发与调试
- 依赖管理：所有依赖在 `requirements.txt`，需用 `pip install -r requirements.txt` 安装。
- 启动服务：通常运行 `app/main.py` 或通过 FastAPI 启动（如 `uvicorn app.main:app --reload`）。
- 配置项集中在 `app/config.py`，如 `TOP_K`、`LLM_CONTEXT_DOCS`。
- 关键参数（如分块大小、重排序 batch_size）可在 API 层或 config 文件调整。

## 项目约定与模式
- 路由与业务逻辑分离，API 层只做参数校验和流程调度，具体处理分散在各功能模块。
- 嵌入、检索、重排序、LLM 生成均有独立模块，便于扩展和替换。
- 元信息（如 source、id、text）在数据流中始终保留，便于追溯和结果解释。
- 错误处理采用 FastAPI 标准异常（如 `HTTPException`），但部分 LLM 生成异常会自动降级到备用模型。

## 重要文件参考
- `app/api.py`：API 路由与主流程。
- `app/embeddings.py`、`app/vectorstore.py`、`app/reranker.py`、`app/pipeline.py`、`app/llm.py`：各功能模块。
- `app/config.py`：全局配置。
- `requirements.txt`：依赖列表。

## 示例模式
- 分块：`chunk_text(text, chunk_size=512, overlap=64)`
- 嵌入：`get_embeddings(chunks)`
- 检索：`search(q_vec, top_k)`
- 重排序：`rerank(query, candidate_texts, batch_size=8)`
- Prompt 构建：`build_rag_prompt(query, context_snippets, max_snippets)`
- LLM 生成：`generate_local(system_prompt, user_prompt)`，异常时自动切换 `generate_openai`

---
如需补充工作流、约定或特殊模式，请在此文档继续完善。