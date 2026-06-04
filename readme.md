基于 LangGraph 构建可评估知识库 Agent 系统，将问答流程拆分为 Query Rewrite、Chroma 向量检索、BM25 关键词检索、RRF 融合、答案生成、引用校验和会话保存等节点。系统支持多用户/多知识库 metadata 隔离、SSE token 级流式输出、LangGraph Checkpoint 会话记忆、Pydantic 工具参数校验和最大工具调用轮数控制，并引入 Ragas 对 faithfulness、context precision、answer relevancy 等指标进行自动化评估。
基于 JWT 实现接口鉴权和用户知识库隔离。
引入 Ragas 对 RAG 系统进行自动化评估，从 faithfulness、context precision、answer relevancy 等维度量化检索与生成质量，并记录每次问答的检索上下文、引用来源和延迟指标。
一开始我只是把向量检索和 BM25 结果拼接去重，但这种方式无法衡量不同检索器的排名贡献。后面我改成 RRF 融合，把 Chroma 和 BM25 的排名统一成融合分数，从而让同时被多个检索器认为相关的 chunk 排得更靠前。
当前版本使用 InMemorySaver 实现会话级状态记忆，后续可替换为 SQLite/Postgres checkpointer 实现持久化 graph state。
我不仅返回引用，还对模型生成的引用编号做后处理校验，避免模型编造不存在的引用来源。
我设置了最大工具调用轮数，防止 Agent 死循环；工具参数用 Pydantic 校验；工具失败后会返回结构化错误信息给模型。
我会记录 query rewrite、检索结果、融合分数、最终 prompt、模型输出、引用、耗时和异常堆栈。
