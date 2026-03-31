# 多模态客服智能体

基于 RAG + 多模态理解 + 思维链拆解 + 幻觉抑制的智能客服系统。

## 项目结构

```
RAGllm/
├── configs/                     # 所有配置集中管理
│   ├── config.yaml              # 主配置（模型、向量库、检索参数等）
│   └── prompts.yaml             # 提示词模板（系统 prompt、RAG prompt、CoT prompt）
│
├── data/                        # 原始数据（只读，不做修改）
│   ├── manuals/                 # 官方提供的说明书原始文件（PDF/DOCX）
│   └── images/                  # 官方提供的说明书配图原始文件
│
├── data_processed/              # 预处理后的结构化数据
│   ├── chunks/                  # 文本切片结果（JSON/JSONL）
│   ├── images_extracted/        # 从 PDF/DOCX 中自动提取的图片
│   ├── image_captions/          # 图片描述（用多模态模型生成的 caption）
│   └── image_text_mapping.json  # 图片与文本段落的关联映射表
│
├── index/                       # 向量索引存储（已有，保留）
│   ├── text_index/              # 文本向量索引（Milvus/FAISS）
│   └── image_index/             # 图片向量索引（CLIP embedding）
│
├── src/                         # 核心源码
│   ├── preprocess/              # 数据预处理模块
│   │   ├── parse_manual.py      # 解析说明书（提取文本 + 提取内嵌图片）
│   │   ├── chunk_text.py        # 文本切分策略（保留图文对应关系）
│   │   └── caption_images.py    # 用多模态模型为图片生成描述文本
│   │
│   ├── indexing/                # 索引构建模块
│   │   ├── build_text_index.py  # 构建文本向量库
│   │   └── build_image_index.py # 构建图片向量库（CLIP embedding）
│   │
│   ├── retrieval/               # 检索模块
│   │   ├── text_retriever.py    # 文本检索器（向量 + BM25 + Rerank）
│   │   ├── image_retriever.py   # 图片检索器（根据文本查相关配图）
│   │   └── multimodal_retriever.py # 多模态融合检索（文本+图片联合返回）
│   │
│   ├── agent/                   # 智能体核心
│   │   ├── agent.py             # 主 Agent 入口（多轮对话管理）
│   │   ├── chain_of_thought.py  # 思维链拆解（复杂问题分步处理）
│   │   ├── hallucination.py     # 幻觉抑制策略（事实核查/置信度过滤）
│   │   └── multimodal_input.py  # 多模态输入处理（解析用户图片+文字）
│   │
│   ├── database.py              # 对话记忆管理（已有，保留扩展）
│   └── utils.py                 # 公共工具函数
│
├── finetune/                    # 模型微调相关
│   ├── data/                    # 微调数据集（多模态对话数据）
│   ├── scripts/                 # 微调脚本
│   │   └── train.py             # 微调入口
│   └── checkpoints/             # 微调模型检查点（.gitignore）
│
├── eval/                        # 评测模块
│   ├── questions/               # 赛题文件（A榜/B榜 400 道题）
│   ├── eval_runner.py           # 批量答题脚本（读入题目 -> Agent 回答 -> 输出结果）
│   ├── eval_metrics.py          # 本地评测指标（自评打分，模拟 LLM 裁判）
│   └── results/                 # 评测结果存档
│
├── submissions/                 # 提交文件
│   └── answer_A.json            # 按赛题格式生成的最终答题文件
│
├── notebooks/                   # 实验用 Jupyter Notebook
│   └── exploration.ipynb        # 数据探索、检索效果调试
│
├── requirements.txt             # 依赖（已有，需扩充多模态相关包）
├── set_env.sh                   # 环境变量（已有）
├── run.py                       # 项目统一入口脚本
└── README.md                    # 项目说明
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置环境变量
source set_env.sh

# 3. 数据预处理（解析说明书、提取图片、生成切片）
python run.py preprocess

# 4. 交互式对话
python run.py chat
如果想更快，可以用 python run.py chat --no-cot --no-hallucination-check 跳过拆解和幻觉检查（ 1 次 API 调用）。

# 5. 批量评测
python run.py eval --questions eval/questions/questions_A.jsonl
```

## 核心能力

- **多模态理解**：支持用户上传图片，自动识别图片内容并辅助检索
- **RAG 知识库**：文本向量检索 + BM25 + FlashRank Rerank，精准匹配说明书内容
- **图文联合返回**：回答中同时包含文字和相关配图
- **思维链拆解**：自动将复杂问题拆分为子问题，逐一作答后汇总
- **幻觉抑制**：基于事实核查的置信度评估，拒绝无依据的编造
- **多轮对话**：Redis 热数据 + MySQL 长期记忆 + 滚动摘要压缩
