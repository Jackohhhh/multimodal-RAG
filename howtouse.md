# rag.py / database.py 说明文档

本文档仅覆盖 `src/rag.py` 与 `src/database.py` 的函数总结、操作指南与使用方式。

## 一、运行前提
- Python 3.10（建议使用现有 `.venv`: `source .venv/bin/activate`）
- 依赖安装：`pip install -r requirements.txt`
- 环境配置：`source set_env.sh`
- 外部服务：
  - MySQL（默认库：`rag_sql`）
  - Redis（默认 `localhost:6379`）
  - Milvus/Zilliz（云端向量库，需要 URI 与 Token）

## 二、关键配置
- `src/rag.py`
  - `DATA_PATH`: 文档目录（默认 `./data/`）
  - `OUTPUT`: 输出文件（默认 `./workspace/ragoutput/answers_ensemble2.md`）
  - `ZILLIZ_URI`, `ZILLIZ_TOKEN`, `COLLECTION_NAME`: Zilliz 连接与集合
  - `TOKENIZERS_PARALLELISM=false`: 避免 HuggingFace fork 警告
  - 环境变量：
    - `OPENAI_API_KEY` 必填
    - `OPENAI_API_BASE` 可选
- `src/database.py`
  - `MYSQL_URL`: MySQL 连接串
  - Redis 使用默认本地连接参数

## 三、操作指南（运行流程）
1) 将需要检索的文档放入 `./data/`（支持 `.pdf/.docx/.doc/.txt`）。
2) 确保 MySQL、Redis、Zilliz 可用。
3) 运行：

```bash
/Users/jack/Documents/RAGllm/.venv/bin/python src/rag.py
```

4) 进入交互式问答，输入问题；输入 `exit` 退出。
5) 每次回答会保存到 `./workspace/ragoutput/answers_ensemble2.md`。

## 四、rag.py 函数与流程总结
**核心流程（概览）**  
加载文件 -> 增量入库 -> 检索 -> 去重 -> 重排 -> 提示词 -> 生成答案 -> 记忆管理 -> 输出保存

**关键函数/类**
- `format_docs(docs)`：将检索结果格式化为上下文字符串，附带来源与评分，并打印调试上下文。
- `format_history(messages)`：把历史消息列表转换成多行文本，供 prompt 使用。
- `UniqueRetriever`：对基础检索器返回结果进行去重（基于 `page_content`）。
- `all_files_in_dir(file_paths)`：按后缀选择加载器，读入文档并写入元数据。
- `get_incremental_vector_store(embeddings)`：连接 Zilliz，读取已入库文件列表，仅对新文件切分并入库。
- `main()`：初始化模型/向量库/检索器/重排器；构建 RAG 链；循环问答并保存结果。

**检索与重排机制**
- 向量检索：`vector_store.as_retriever(k=10)`
- 去重：`UniqueRetriever`
- 重排：`FlashrankRerank(top_n=5)` + `ContextualCompressionRetriever`
- 仅使用检索到的上下文回答（缺失则返回不知道）

## 五、database.py 函数与流程总结
**数据模型**
- `MySQLChatLog`：记录完整对话（每条一行）。
- `MySQLChatSummary`：每个 session 一个摘要（`session_id` 主键）。

**MemoryManager（分层记忆）**
- `get_context_messages()`：从 MySQL 取摘要 + 从 Redis 取热记录，拼成消息列表。
- `save_context(human_text, ai_text)`：双写 MySQL/Redis；超过阈值触发压缩。
- `_push_to_redis(role, content)`：写入热数据并设置 24h 过期。
- `_compress_old_messages(popped_human, popped_ai)`：用 LLM 更新摘要并落库。

## 六、使用方法与常见调整
- **文档更新后不生效**：当前只增量入库，同名文件不会重建；可改 `drop_old=True` 或手动删除集合后重建。
- **检索数量**：改 `search_kwargs["k"]` 与 `FlashrankRerank(top_n)`。
- **切分参数**：改 `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)`。
- **记忆窗口**：改 `MemoryManager(..., retain_n_turns=3)`。
- **输出格式**：改 `format_docs` 或 prompt 模板。
