# RAG 前端项目

这是一个基于 Vue 3 + Vite 的 RAG API 前端界面。

## 安装设置

1.  **安装 Node.js**：确保您已安装 Node.js（推荐 v18+ 版本）。
2.  **安装依赖**：
    ```bash
    cd frontend
    npm install
    ```

## 开发指南

1.  **启动后端**：
    确保 FastAPI 后端正在运行（通常在端口 8000 或 8001）。
    ```bash
    # 在项目根目录下运行
    uvicorn app.main:app --reload
    ```

2.  **启动前端**：
    ```bash
    # 在 frontend 目录下运行
    npm run dev
    ```

3.  **访问**：
    打开终端显示的 URL（通常是 `http://localhost:5173`）。

## 功能特性

- **调试控制台**：执行查询并查看详细的执行分解：
    - 各阶段执行耗时（Embedding、Search、Rerank、Generation）。
    - 检索到的文本片段（初排候选 vs 重排后上下文）。
    - 最终生成的回答。
