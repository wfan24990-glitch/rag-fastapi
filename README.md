# RAG FastAPI Service

这是一个基于 FastAPI 构建的高性能 RAG (Retrieval-Augmented Generation) 服务。它提供了一套完整的从文本入库到智能问答的解决方案，集成了先进的文本嵌入、向量检索、重排序 (Reranking) 以及大语言模型 (LLM) 生成能力。

## ✨ 核心特性

- **高性能架构**: 基于 FastAPI 异步框架，支持高并发请求。
- **先进的检索链路**:
  - **Embedding**: 默认集成 `BAAI/bge-small-zh-v1.5`，支持中文语义向量化。
  - **Vector Store**: 使用 FAISS 进行高效的向量索引和检索。
  - **Reranking**: 集成 `BAAI/bge-reranker-base` 对检索结果进行语义重排序，显著提升相关性。
- **智能问答**:
  - 自动构建包含上下文引用的 Prompt。
  - 支持本地 LLM 接入。
  - **自动降级策略**: 当本地 LLM 不可用或出错时，自动切换至 OpenAI API 作为备用。
- **异步处理**: 文本入库 (`/ingest`) 采用后台任务处理，不阻塞主线程。

## 🏗 系统架构

数据流向如下：

1.  **入库 (Ingestion)**:
    `文本输入` -> `分块 (Chunking)` -> `Embedding 模型` -> `向量 (Vectors)` -> `FAISS 索引` & `元数据存储`

2.  **问答 (Query)**:
    `用户问题` -> `Embedding 模型` -> `向量检索 (Top-K)` -> `Reranker 重排序` -> `构建 Prompt (含上下文)` -> `LLM 生成` -> `最终答案`

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- (可选) NVIDIA GPU + CUDA (用于加速 Embedding 和 Reranking 模型推理)

### 2. 安装依赖

```bash
git clone https://github.com/wfan24990-glitch/rag-fastapi.git
cd rag-fastapi
pip install -r requirements.txt
```

### 3. 配置环境

在项目根目录创建 `.env` 文件（参考以下配置）：

```ini
# LLM 配置 (本地/主模型)
LLM_API_KEY=your_local_llm_key
LLM_PROVIDER=local
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=your-local-model-name

# OpenAI 配置 (备用模型)
OPENAI_API_KEY=sk-xxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# 模型路径配置 (可选，默认自动下载)
# EMBEDDING_MODEL_PATH=BAAI/bge-small-zh-v1.5
# RERANKER_MODEL=BAAI/bge-reranker-base

# 向量库路径
FAISS_INDEX_PATH=data/faiss_index.bin

# 检索参数
TOP_K=20
LLM_CONTEXT_DOCS=5
```

### 4. 启动服务

```bash
python app/main.py
# 或者使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

服务启动后，API 文档地址: `http://localhost:8001/docs`

## 📖 API 使用指南

### 1. 文本入库 (`/ingest`)

将文本数据添加到知识库中。

**请求:**
```bash
curl -X POST "http://localhost:8001/ingest" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "FastAPI 是一个用于构建 API 的现代、快速（高性能）的 web 框架，使用 Python 3.6+ 并基于标准的 Python 类型提示。",
           "source": "fastapi_intro"
         }'
```

**响应:**
```json
{
  "status": "processing",
  "ingested_chunks_count": 1,
  "message": "Ingestion started in background"
}
```

### 2. 智能问答 (`/query`)

基于知识库回答问题。

**请求:**
```bash
curl -X POST "http://localhost:8001/query" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "FastAPI 是什么？",
           "top_k": 10
         }'
```

**响应:**
```json
{
  "answer": "FastAPI 是一个用于构建 API 的现代、快速（高性能）的 web 框架...",
  "sources": [
    {
      "text": "FastAPI 是一个用于构建 API 的现代...",
      "score": 0.98,
      "source": "fastapi_intro",
      "id": 0
    }
  ]
}
```

### 3. 健康检查 (`/status`)

```bash
curl http://localhost:8001/status
```

## 📂 项目结构

```
rag-fastapi/
├── app/
│   ├── api.py           # API 路由定义
│   ├── config.py        # 配置加载
│   ├── embeddings.py    # Embedding 模型封装
│   ├── llm.py           # LLM 调用逻辑 (含 Fallback)
│   ├── llm_client.py    # 通用 LLM 客户端
│   ├── main.py          # 程序入口
│   ├── pipeline.py      # RAG Prompt 构建
│   ├── reranker.py      # Reranker 模型封装
│   ├── vectorstore.py   # FAISS 向量库管理
│   └── utils/
│       └── chunker.py   # 文本分块工具
├── data/                # 存放向量索引文件 (faiss_index.bin)
├── requirements.txt     # 项目依赖
└── README.md            # 项目文档
```

## 🛠 技术栈

- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss)
- **ML Models**: [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) (PyTorch)
- **LLM Integration**: Custom Client + OpenAI SDK

## 📝 License

MIT License
