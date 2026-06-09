# 本次代码修改说明

本次修改针对以下 5 个问题：

1. README 中提到 JWT 鉴权，但接口未接入。
2. tool_node 有基础实现，但工具调用链路不完整。
3. kb_id 隔离没有完整贯通。
4. requirements.txt 不完整。
5. 当前切片是固定长度切片，不是语义切片。

## 主要修改

### 1. JWT 鉴权

新增：

- `backend/auth/dependencies.py`
- `/auth/token` 开发 token 接口

修改：

- `/upload` 需要 JWT
- `/chat` 需要 JWT
- `/chat/stream` 需要 JWT

普通 HTTP 请求通过：

```http
Authorization: Bearer <token>
```

流式请求由于浏览器 `EventSource` 不能设置自定义 header，所以允许：

```text
/chat/stream?access_token=<token>&kb_id=default&question=...
```

### 2. 工具调用链路

修改：

- `backend/graph/nodes.py`
- `backend/graph/workflow.py`

现在 `classify_question` 会在识别到计算问题时写入：

- `route="tool"`
- `tool_name="calculator"`
- `tool_args={...}`

然后进入：

```text
classify_question -> tool_node -> generate_answer -> validate_answer -> save_message
```

### 3. kb_id 隔离

修改：

- `backend/app.py`
- `backend/db/sqlite.py`
- `backend/retrieval/chroma_store.py`
- `backend/retrieval/bm25_store.py`
- `backend/graph/state.py`
- `backend/graph/nodes.py`

现在上传、Chroma 检索、BM25 检索、会话、消息、评估记录都带有 `kb_id`。

检索时使用：

```text
user_id + kb_id
```

共同过滤，避免不同用户或不同知识库互相串数据。

### 4. requirements.txt

新增：

- `requirements.txt`

包含 FastAPI、LangGraph、LangChain、Chroma、Ollama、HuggingFace Embeddings、文档解析、RAGAS 评估、测试等依赖。

### 5. 语义切片

修改：

- `backend/documents/splitter.py`

现在 `split_text` 默认使用：

```python
splitter_type="semantic"
```

优先调用 `langchain_experimental.text_splitter.SemanticChunker` 做基于 embedding 的语义切片；如果环境缺少依赖或 embedding 模型不可用，会自动退回到 `RecursiveCharacterTextSplitter`，尽量按照标题、段落、句子边界切分，避免接口直接失败。
