# 尝尝咸淡 · 中文食谱 RAG + Agent 问答系统

> 一个从 **RAG 基础管线**逐步演化为 **Agentic RAG 多智能体系统**的完整项目。
> 输入一句"今天吃什么"，由主管 Agent 调度"菜谱专家"和"推荐专家"从 323 道菜的本地知识库中检索并生成回答。

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [数据格式规范](#数据格式规范)
- [配置说明](#配置说明)
- [设计要点](#设计要点)
- [已知限制](#已知限制)
- [后续计划](#后续计划)

---

## 项目简介

面向中文菜谱领域的检索增强生成（RAG）问答系统，并在此之上构建了完整的 Agent 能力：

- **底层**：经典四段式 RAG 管线（数据准备 → 索引构建 → 检索优化 → 生成集成），基于 LangChain 0.3 + FAISS + bge-small-zh-v1.5 + 可插拔 LLM（Moonshot / DeepSeek / OpenAI）
- **上层**：LangGraph 编排的 Agent 体系——Fast/Slow 双路径路由、ReAct 工具调用循环、检索自纠错、Reflexion 生成反思、主管-工人多智能体协作

目标是解决"今天吃什么"的选择困难症：既支持"红烧肉怎么做"的具体问答，也支持"推荐几个简单的素菜"这类推荐/筛选需求。

## 功能特性

**RAG 底座**
- Markdown 结构感知分块（父子双层结构：小块精准检索、整篇菜谱生成）
- 元数据自动增强（菜品分类 / 菜名 / 难度 ★ 级）
- 混合检索：向量语义检索 + BM25 关键词检索，RRF 融合重排
- 元数据过滤（分类 / 难度）+ FAISS 索引本地缓存复用

**Agent 能力**
- **Fast/Slow 分层路由**：简单问题走标准管线（快、省 token），复杂/模糊问题进入 Agent 循环
- **工具调用**：5 个 Function Tools（菜谱搜索 / 整谱获取 / 分类筛选 / 分类列表 / 查询重写）
- **检索自纠错**：检索无果时自动调用 `rewrite_query` 重写查询再查
- **Reflexion 生成反思**：回答生成后自我质量检查，不合格则按意见修订
- **多智能体编排**：主管 Agent 委派"菜谱 Agent"（做法/食材/步骤）与"推荐 Agent"（推荐/筛选），Worker 各自独立 ReAct 循环、工具角色隔离
- **多轮记忆**：对话历史 + 窗口裁剪（最近 6 轮、超长截断）
- **流式输出**：标准管线支持流式

## 系统架构

```
用户输入
   │
   ▼
┌──────────── RecipeRAGSystem.ask_question() ────────────┐
│  Router(LLM 判断复杂度)                                 │
│   ├─ "fast"  → 标准管线（列表/详细/一般三类回答）        │
│   └─ "react" → 多智能体编排                             │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────────────┐
        ▼ Fast Path                          ▼ Slow Path
┌────────────────────────┐   ┌─────────────────────────────────────┐
│ 查询路由(list/detail/  │   │ Supervisor(主管 LLM)                 │
│   general)             │   │  ├─ delegate_to_recipe_worker        │
│ → 查询重写             │   │  └─ delegate_to_recommender_worker   │
│ → 混合检索 + RRF       │   │ Worker = 独立 ReAct 循环:            │
│ → 元数据过滤           │   │   agent ⇄ tools                     │
│ → 还原父文档           │   │   ├─ 检索自纠错(rewrite_query)       │
│ → 模板生成/列表        │   │   └─ Reflexion 生成反思              │
└────────────────────────┘   └─────────────────────────────────────┘
```

**数据流全景**：

```
cook/ 菜谱.md
  → ① 数据准备：加载 → 元数据增强(分类/菜名/难度) → Markdown 标题分块(父子结构)
  → ② 索引构建：bge-small-zh-v1.5 向量化 → FAISS → 缓存到 vector_index/
  → ③ 检索优化：向量 ⊕ BM25 → RRF 融合 → 元数据过滤
  → ④ 生成集成：MoonshotChat + 查询路由/重写 + 多模板生成
  → ⑤ Agent 层：工具封装 → ReAct 循环 → 反思 → 主管调度
```

## 目录结构

```
code/C8/
├── main.py                        # 入口：RecipeRAGSystem 编排 + 交互式 CLI
├── config.py                      # RAGConfig 配置类（路径/模型/检索/生成参数）
├── requirements.txt               # 依赖清单
├── .env                           # LLM API Key（按 provider 选择，不提交到仓库）
├── cook/                          # 菜谱数据源（323 个 Markdown）
│   └── dishes/<分类>/<菜名>/<菜名>.md
├── vector_index/                  # FAISS 索引缓存（首次构建后自动生成）
└── rag_modules/
    ├── __init__.py                # 模块统一导出
    ├── data_preparation.py        # ① 数据准备：加载/元数据增强/Markdown 分块
    ├── index_construction.py      # ② 索引构建：Embedding + FAISS 持久化
    ├── retrieval_optimization.py  # ③ 检索优化：混合检索 + RRF + 元数据过滤
    ├── generation_integration.py  # ④ 生成集成：MoonshotChat + 路由/重写/多模板
    ├── agent_tools.py             # ⑤ Agent 工具层：RAG 能力封装为 Function Tools
    ├── agent_graph.py             # ⑥ 单 Agent：LangGraph ReAct 循环 + Reflexion
    └── agent_supervisor.py        # ⑦ 多智能体：主管-工人编排（菜谱/推荐专家）
```

## 快速开始

**环境要求**：Python 3.10+（已验证 3.14 可运行；需联网下载嵌入模型与调用 Kimi API）

```bash
# 1. 安装依赖（已在 requirements.txt 中剔除 Python 3.14 无法安装的包）
pip install -r requirements.txt

# 2. 配置 API Key（项目根目录创建 .env，按 provider 选一行）
MOONSHOT_API_KEY=sk-xxxxxxxx          # provider=moonshot 时
# DEEPSEEK_API_KEY=sk-xxxxxxxx        # provider=deepseek 时（同时改 config 的 llm_model 为 deepseek-chat）
# OPENAI_API_KEY=sk-xxxxxxxx          # provider=openai 时

# 3. 运行（须在 code/C8 目录下执行，数据路径为相对路径 ./cook）
python main.py
```

首次运行会自动：
1. 下载中文嵌入模型 `BAAI/bge-small-zh-v1.5`（约 100MB，国内网络可设置 `HF_ENDPOINT=https://hf-mirror.com` 加速）
2. 构建 FAISS 索引并缓存到 `vector_index/`（之后启动直接加载缓存）

运行后进入交互式问答，示例问题：

```
您的问题: 宫保鸡丁怎么做
您的问题: 推荐几个简单的素菜
您的问题: 清蒸鲈鱼需要什么食材
您的问题: 有什么汤品适合冬天
```

## 数据格式规范

数据位于 `cook/dishes/<分类>/<菜名>/<菜名>.md`，遵循约定：

```markdown
# 银耳莲子粥的做法          ← 一级标题（菜名）
预估烹饪难度：★★★★        ← ★ 数量 = 难度（1~5）
## 必备原料和工具           ← 二级标题（食材）
## 计算                     ← 二级标题（用量）
## 操作                     ← 二级标题（步骤）
## 附加内容                 ← 二级标题（技巧）
```

- **分类**：由目录名映射（`meat_dish→荤菜`、`vegetable_dish→素菜`、`soup→汤品`、`dessert→甜品`、`breakfast→早餐`、`staple→主食`、`aquatic→水产`、`condiment→调料`、`drink→饮品`），未覆盖目录归为"其他"
- **难度**：从正文 `★` 数量提取（1~5 档）
- **分块**：按 `# / ## / ###` 三层标题切块，子块携带父文档 ID（`parent_id`），检索命中子块后还原整篇菜谱

## 配置说明

`config.py` 中的 `RAGConfig` 数据类：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `data_path` | `./cook` | 菜谱数据目录（相对项目根） |
| `index_save_path` | `./vector_index` | FAISS 索引缓存目录 |
| `embedding_model` | `BAAI/bge-small-zh-v1.5` | 嵌入模型（HuggingFace） |
| `llm_provider` | `moonshot` | LLM 提供商：`moonshot` / `deepseek` / `openai` |
| `llm_model` | `kimi-k2.6` | 对话模型（DeepSeek 用 `deepseek-chat`，支持工具调用） |
| `llm_base_url` | 空 | API 地址；为空用各提供商默认（deepseek→api.deepseek.com，openai→api.openai.com/v1） |
| `top_k` | `3` | 检索返回候选数 |
| `temperature` | `0.1` | 生成温度（低 = 更稳定） |
| `max_tokens` | `2048` | 最大生成 token 数 |

## 设计要点

1. **父子文档结构**：小块（标题章节）精准召回，整篇菜谱作为生成上下文，用"命中子块数"对父文档去重排序——检索精度与上下文完整性的经典权衡
2. **混合检索 + RRF**：向量（语义）与 BM25（关键词，对菜名/食材专名友好）双路召回，Reciprocal Rank Fusion 融合
3. **路由驱动的差异化生成**：`list`（只要菜名）类查询不调 LLM 直接拼列表，零 token 消耗；`detail` 走分步指导模板
4. **Fast/Slow 分层**：简单问题走确定性管线（快、稳、省），复杂/失败场景才进 Agent 循环（灵活、自适应）
5. **双闭环自纠错**：检索层（`rewrite_query` 重写再查）+ 生成层（Reflexion 自我反思修订）
6. **多智能体角色隔离**：每个 Worker 只绑定自己的工具子集，主管仅持有委派工具，杜绝越权调用
7. **防御性设计**：Router 解析失败保守走 Fast Path、ReAct 步数上限、委派次数上限、反思失败保留原回答——所有循环都有闸门

## 已知限制

- 需要 LLM 提供商 API Key（每次问答约 2~5 次 LLM 调用；免费账号通常有 RPM 限流，如 Moonshot 每分钟 3 次）
- Agent 路径（ReAct / 多智能体）暂不支持流式输出，流式走标准管线
- 上下文拼接有 2000 字符预算，超长菜谱可能被截断
- 分类映射未覆盖 `semi-finished` / `template` 目录（归为"其他"）
- 元数据过滤为纯关键词匹配，检索失败时暂不回退到无过滤检索
- 数据加载仅支持 Markdown（目录结构 + 标题 + ★ 约定）

## 后续计划

- [ ] 多智能体路径流式输出支持
- [ ] 检索失败自动回退（Agent 已有自纠错，Fast Path 补齐）
- [ ] 语义分块 / 重排序模型（cross-encoder）提升检索质量
- [ ] 评估集 + 指标（Recall@k / 忠实度）与对比实验（Fast vs ReAct vs 多智能体）
- [ ] pytest 单元测试与 CI
- [ ] 多格式数据源加载（PDF / TXT / JSON）

---

*技术栈：LangChain 0.3 · LangGraph · FAISS · sentence-transformers · BM25 · 多LLM提供商（Moonshot / DeepSeek / OpenAI）*
