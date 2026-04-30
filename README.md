# 多模态 RAG 客服智能体

一个面向产品说明书与电商售后场景的多模态 RAG（Retrieval-Augmented Generation）问答系统。项目将说明书文本和配图标记统一处理为可检索知识库，支持交互式客服问答、批量赛题评测、图片 ID 关联返回、问题拆解和幻觉检查。

> 当前仓库默认使用 FAISS 本地向量索引、`BAAI/bge-m3` 文本向量模型，以及 OpenAI 兼容接口的大语言模型。模型、向量库和检索策略均可在 `configs/config.yaml` 中调整。

## 目录

- [核心特性](#核心特性)
- [系统流程](#系统流程)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [数据格式](#数据格式)
- [常用命令](#常用命令)
- [评测与提交](#评测与提交)
- [故障排查](#故障排查)
- [开发说明](#开发说明)

## 核心特性

- **说明书知识库 RAG**：将产品手册切分为带来源、图片 ID 和检索分数的文档片段，支持向量召回、BM25 混合召回、去重和可选 FlashRank 精排。
- **图文关联返回**：预处理时把正文中的 `<PIC>` 占位符按顺序映射为 `[IMG:图片ID]`，回答后自动提取关联图片 ID。
- **多模态输入理解**：用户上传图片时，系统先用多模态 LLM 生成图片描述，再拼接到文本问题中参与检索。
- **客服兜底能力**：对退款、物流、发票、售后等通用客服问题，可绕过说明书检索，直接生成稳妥客服回复。
- **复杂问题拆解**：可选启用 CoT 风格问题拆解，对多问题输入逐条检索与合并回答。
- **幻觉抑制**：可选启用 grounding 检查，判断回答是否被上下文支持，降低无依据编造。
- **批量评测导出**：支持 JSONL/CSV 题库输入，自动生成结果文件和提交文件。

## 系统流程

```text
data/manuals/*.txt
      │
      ▼
解析 [text, [image_ids]]，将 <PIC> 映射为 [IMG:id]
      │
      ▼
文本切分，保留 source 与 related_images 元数据
      │
      ▼
BAAI/bge-m3 生成文本向量，写入 FAISS/Milvus
      │
      ▼
用户问题 / 用户图片描述
      │
      ▼
向量检索 + BM25 混合召回 + 去重 + 可选 Rerank
      │
      ▼
LLM 基于上下文生成回答，保留 [IMG:id]
      │
      ▼
输出回答文本与 image_ids
```

## 项目结构

```text
multimodal-RAG/
├── configs/
│   ├── config.yaml              # 模型、向量库、切分和检索参数
│   └── prompts.yaml             # RAG、客服兜底、拆解、合并、幻觉检查提示词
├── data/
│   └── manuals/                 # 原始说明书数据，内容为 [text, [image_ids]]
├── data_processed/
│   └── chunks/chunks.jsonl      # 预处理后的文本切片
├── docs/
│   └── multimodal_pipeline_guide.md
├── eval/
│   ├── questions/               # 批量评测题库，支持 CSV / JSONL
│   ├── results/                 # 批量答题结果
│   ├── eval_runner.py           # 批量答题脚本
│   └── eval_metrics.py          # LLM 裁判式本地评测
├── index/
│   └── text_index/faiss/        # FAISS 本地索引，预处理生成
├── src/
│   ├── agent/                   # 主 Agent、问题拆解、幻觉检查、多模态输入
│   ├── indexing/                # 文本/图片 caption 索引辅助模块
│   ├── preprocess/              # 说明书解析、文本切分、图片 caption 生成
│   ├── retrieval/               # 向量检索、BM25 混合召回、图片 ID 提取
│   ├── utils.py
│   └── vector_store.py          # FAISS / Milvus 构建与加载工厂
├── submissions/                 # 提交文件输出目录
├── requirements.txt
├── run.py                       # 统一命令行入口
└── README.md
```

## 快速开始

### 1. 准备 Python 环境

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

请创建或修改本地环境变量文件，并用占位值替换为自己的密钥。不要把真实 API Key 提交到 Git。

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.deepseek.com/v1"
```

加载环境变量：

```bash
source set_env.sh
```

### 3. 构建知识库索引

```bash
python run.py preprocess
```

该命令会读取 `configs/config.yaml` 中的 `data.manuals_path`，解析 `data/manuals` 下的说明书，生成 `data_processed/chunks/chunks.jsonl`，并将向量索引写入 `index/text_index/faiss`。

如果是首次使用 `BAAI/bge-m3`，需要确保模型已在本机 Hugging Face 缓存中。项目启动时默认离线加载 Hugging Face 模型；首次下载可临时执行：

```bash
HF_HUB_OFFLINE=0 python run.py preprocess
```

### 4. 启动交互式客服

```bash
python run.py chat
```

启用问题拆解和幻觉检查：

```bash
python run.py chat --cot --hallucination-check
```

交互中输入 `exit` 或 `quit` 退出。

## 配置说明

主要配置位于 `configs/config.yaml`。

```yaml
llm:
  model: "deepseek-chat"
  provider: "openai"
  temperature: 0.8

embeddings:
  model_name: "BAAI/bge-m3"

vector_store:
  type: "faiss"
  index_path: "./index/text_index/faiss"

retrieval:
  use_hybrid_bm25: true
  use_rerank: false
  search_k: 15
  vector_top_k: 6
  bm25_top_k: 5
  rag_relevance_threshold: 0.4
```

常见可调项：

- `llm`：主回答模型，支持 OpenAI 兼容接口或 `google_genai`。
- `multimodal_llm`：图片理解模型，用于用户上传图片描述或图片 caption 生成。
- `vector_store.type`：默认 `faiss`，也可按 `src/vector_store.py` 支持方式配置为 `milvus`。
- `retrieval.use_hybrid_bm25`：是否启用向量 + BM25 混合召回。
- `retrieval.use_rerank`：是否启用 FlashRank 精排。
- `retrieval.rag_relevance_threshold`：RAG 命中阈值，低于阈值时触发无依据或客服兜底逻辑。

提示词集中维护在 `configs/prompts.yaml`，包括 RAG 主提示词、客服兜底、英文检索改写、问题拆解、答案合并和幻觉检查。

## 数据格式

当前预处理入口期望 `data/manuals` 下的 `.txt` 或 `.json` 文件内容为 JSON 格式：

```json
[
  "说明书正文。第一个配图位置 <PIC>，第二个配图位置 <PIC>。",
  ["Manual01_0", "Manual01_1"]
]
```

处理规则：

- 文本中的第 N 个 `<PIC>` 会对应 `image_ids` 中的第 N 个图片 ID。
- 预处理后 `<PIC>` 会被替换为 `[IMG:Manual01_0]` 这类标记。
- 文本切片时会保留 `source` 和 `related_images`，方便回答时返回图片 ID。
- 如果 `<PIC>` 数量与图片 ID 数量不一致，程序会按较少的一方匹配并输出警告。

## 常用命令

查看命令帮助：

```bash
python run.py --help
python run.py chat --help
python run.py eval --help
```

使用自定义配置：

```bash
python run.py --config configs/config.yaml --prompts configs/prompts.yaml chat
```

重新生成知识库：

```bash
python run.py preprocess
```

启动交互式问答：

```bash
python run.py chat
```

开启调试检索日志：

```bash
export RAG_VERBOSE_RETRIEVAL=1
python run.py chat
```

## 评测与提交

批量评测支持 CSV 和 JSONL 题库。题目至少包含 `id`、`question` 字段，CSV 可选 `image_path` 字段。

```bash
python run.py eval --questions eval/questions/question_public.csv
```

显式指定输出路径：

```bash
python run.py eval \
  --questions eval/questions/question_public.csv \
  --output eval/results/question_public_results.jsonl \
  --submission submissions/question_public_submission.csv
```

启用拆解与幻觉检查：

```bash
python run.py eval \
  --questions eval/questions/question_public.csv \
  --cot \
  --hallucination-check
```

使用 LLM 裁判评测已有结果：

```bash
python eval/eval_metrics.py \
  --results eval/results/question_public_results.jsonl \
  --questions eval/questions/question_public.csv \
  --output eval/results/question_public_evaluated.jsonl
```

## 故障排查

### FAISS 索引不存在

如果启动 chat 或 eval 时提示 `FAISS 索引不存在`，请先运行：

```bash
python run.py preprocess
```

### Hugging Face 模型无法加载

项目默认设置 `HF_HUB_OFFLINE=1`，适合模型已缓存的环境。首次运行可关闭离线模式：

```bash
HF_HUB_OFFLINE=0 python run.py preprocess
```

### FlashRank 初始化失败

当 `retrieval.use_rerank: true` 时，首次运行可能需要下载 FlashRank ONNX 模型。若网络或 SSL 受限，可先关闭精排：

```yaml
retrieval:
  use_rerank: false
```

也可以按报错提示手动下载模型到 `models/flashrank`。

### API Key 或模型提供商错误

- OpenAI 兼容模型使用 `OPENAI_API_KEY` 和可选 `OPENAI_API_BASE`。
- Gemini 使用 `GOOGLE_API_KEY` 或 `GEMINI_API_KEY`，并将 `provider` 配置为 `google_genai`。
- 不要在公开仓库中提交 `set_env.sh` 或任何真实密钥。

## 开发说明

- 新增说明书数据后，重新运行 `python run.py preprocess` 生成切片和索引。
- 检索相关逻辑主要在 `src/retrieval/text_retriever.py` 与 `src/retrieval/hybrid_retriever.py`。
- Agent 主流程在 `src/agent/agent.py`，包括客服兜底、RAG、问题拆解、英文检索改写和幻觉检查。
- 更详细的多模态处理说明见 `docs/multimodal_pipeline_guide.md`。该文档包含历史方案说明，若与当前配置不一致，以 `configs/config.yaml` 和 `src/vector_store.py` 为准。

## License

当前仓库尚未声明开源许可证。如需公开发布，建议补充 `LICENSE` 文件并在此处说明授权范围。
