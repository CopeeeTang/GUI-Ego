# Streaming Memory Agent — 实施路线图

> 日期: 2026-03-03 (updated: 2026-03-03 R3)
> 状态: 规划阶段 → 设计确认 → 数据/Benchmark 策略确认
> 前置调研: `docs/survey/streaming-benchmark-survey.md`, `docs/memory/2026-02-26-change-driven-memory-design.md`
> 会话历史: `docs/history/memory/round_1.md`, `round_2.md`
> 项目路径: `/home/v-tangxin/GUI/streaming-agent/` (独立项目)

---

## 研究定位

**填补空白**: Proactive + Agent + Trainable（目前该格子为空）

| | Non-Agent | Agent |
|--|-----------|-------|
| **Proactive** | 16 个模型 (拥挤) | StreamAgent (唯一, training-free) |
| **Responsive** | 6 个模型 | VITAL, EventMemAgent, VideoARM |

方法论: Context Engineering for Streaming Video Agents
核心贡献: Change-Driven Segment Memory（变化驱动的结构化记忆）

### 对标竞争者

| 竞争者 | OVO-Bench | StreamingBench | 特点 | 我们的差异化 |
|--------|-----------|----------------|------|-------------|
| EventMemAgent | 60.75 | 77.00 (RT) | 像素级事件分段, SFT+GRPO | 语义级 CLIP Δ, Proactive 能力 |
| StreamAgent | 49.4 | 74.28 | Training-free, 预期式规划 | 可训练, 结构化记忆, 多工具 |
| StreamBridge | 71.30 | 77.04 | 0.5B+7B 双模型, Non-Agent | Agent 架构, 工具调用, Proactive |
| StreamForest | — | 77.3 | 事件记忆森林, Responsive | Proactive, 工具增强, 可训练 |

---

## 核心架构

### 三层记忆 + 三级快慢系统

```
Layer 0: Task Context (全局不变上下文，类比 CLAUDE.md)
Layer 1: Segment Memory (CLIP Δ 变化驱动切片 + summary + entity snapshot)
Layer 2: Visual Buffer (近期帧滑动窗口)

Tier 0 (5ms/帧):  CLIP Δ 信号 → skip 86% / store 11% / trigger 3%
Tier 1 (200ms/段): VLM 生成 summary + entity → 纯计算 transition → 评估 proactive
Tier 2 (1-2s/需):  完整 Agent 推理 + ≤2 轮工具调用 (ReAct)
```

### 记忆数据结构（三种信息类型）

#### A. 段内摘要 (Intra-Segment) — "用户正在看着水壶"
```python
@dataclass
class Segment:
    seg_id: int
    t_start: float
    t_end: float
    summary: str                          # VLM 一帧生成: "用户用菜刀切洋葱"
    keyframe: Image                       # CLIP 特征最接近段均值的 1 帧
    entity_snapshot: dict[str, EntityState]  # VLM 同帧生成: {洋葱: 正在被切, 菜刀: 使用中}
    embedding: np.ndarray                 # summary 的语义向量 (all-MiniLM-L6-v2)
    clip_delta_mean: float                # 段内平均变化量
    clip_delta_at_start: float            # 段起始处的 Δ (边界强度)
```

#### B. 段间转换 (Inter-Segment) — "用户从炒菜 → 拿调料"
```python
@dataclass
class Transition:
    from_seg: int
    to_seg: int
    entity_changes: dict[str, tuple[str, str]]  # {"菜刀": ("使用中", "放在砧板上")}
    t_boundary: float                            # 段落边界时间戳 (供 diff 使用)
    clip_delta: float                            # 边界处的 CLIP Δ 值
    # 注意: 不需要 VLM 调用，纯计算 entity diff
    # 自然语言描述在 timeline() 调用时用模板拼接生成
```

#### C. 实体状态 (Entity State) — "水壶: 冷水 → 加热中 → 沸腾"
```python
@dataclass
class EntityState:
    state: str      # "加热中"
    location: str   # "灶台"

class SimpleEntityTracker:
    """Phase 0 简单版: 仅拼接历史，不做名称归一化"""
    history: dict[str, list[tuple[int, EntityState]]]  # entity_name → [(seg_id, state)]

    def update(self, seg_id: int, entities: dict[str, EntityState]):
        for name, state in entities.items():
            self.history[name].append((seg_id, state))

    def track(self, entity: str) -> list[tuple[int, EntityState]]:
        # Phase 0: 简单子串模糊匹配
        matches = [k for k in self.history if entity in k or k in entity]
        return self.history.get(matches[0], []) if matches else []
```

### Tier 1 处理流程

```
段落边界确定 (Tier 0 检测到 CLIP Δ > θ_high):
  │
  ├─ Step A: 选关键帧 (无 VLM, ~1ms)
  │    从当前段帧缓冲中选 CLIP 特征最接近段内均值的 1 帧
  │
  ├─ Step B: VLM 生成 summary + entities (一次调用, ~150ms)
  │    输入: 1 张关键帧 + 结构化 prompt
  │    输出: {summary: "用户用菜刀切洋葱",
  │           entities: {洋葱: {state: "正在被切", location: "砧板"}}}
  │    Phase 0 先尝试一次调用两段输出，entity 质量差再拆开
  │
  ├─ Step C: 计算 transition (无 VLM, ~1ms)
  │    diff(prev_seg.entity_snapshot, cur_seg.entity_snapshot)
  │
  ├─ Step D: 更新 Entity Tracker (~1ms)
  │
  └─ Step E: 评估 Proactive 触发条件
       ├─ 不触发 → 等下一段
       └─ 触发 → 进入 Tier 2 Agent 推理
```

### 工具集（契合 Streaming 任务）

| 工具 | 签名 | 延迟 | 对应 Benchmark 任务 |
|------|------|------|---------------------|
| `recall` | `recall(query, top_k=5)` | ~50ms | OVO Backward Tracing |
| `observe` | `observe(target="current" \| seg_id, question=None)` | ~300ms | OVO Realtime Understanding |
| `diff` | `diff(seg_a, seg_b)` 或 `diff(mode="recent")` | ~1ms(预计算) / ~500ms(跨段) | ESTP Proactive + OVO Forward |
| `timeline` | `timeline(last_n=5)` | ~10ms | 全局概览 (streaming 只看最近) |
| `track` | `track(entity)` | ~10ms | 实体状态追踪 (Phase 0 简化版) |

#### 工具与任务类型的映射
```
Backward Tracing (回溯: "刚才发生了什么？"):
  → recall(query) [核心] + observe(seg_id) [辅助]

Realtime Understanding (实时: "现在在发生什么？"):
  → observe("current", question) [核心]

Forward Active Responding (前瞻/主动: "需要提醒用户什么？"):
  → diff("recent") [触发] + recall(task_context) [内容生成]
```

#### 工具实现概要
```python
def recall(query: str, top_k: int = 5) -> list[Segment]:
    """语义检索历史段落"""
    query_emb = encode(query)
    return segment_store.search_by_embedding(query_emb, top_k)

def observe(target: str = "current", question: str = None) -> str:
    """VLM 精细感知 — 支持 'current' 看当前帧 (Realtime 核心)"""
    if target == "current":
        frame = visual_buffer.get_latest_frame()
    else:
        segment = segment_store.get(int(target))
        frame = segment.keyframe
    return vlm.describe(frame, question)

def diff(seg_a: int = None, seg_b: int = None, mode: str = "explicit") -> DiffResult:
    """跨段对比 — mode='recent' 返回最近两段的预计算 transition"""
    if mode == "recent":
        return transition_store.get_latest()
    if seg_b == seg_a + 1:
        return transition_store.get(seg_a, seg_b)  # 相邻段: 预计算
    return compute_entity_diff(
        segment_store.get(seg_a).entity_snapshot,
        segment_store.get(seg_b).entity_snapshot
    )

def timeline(last_n: int = 5) -> list[dict]:
    """最近 N 段概览 (streaming 不需要全局)"""
    segments = segment_store.get_recent(last_n)
    result = []
    for seg in segments:
        result.append({"seg_id": seg.seg_id, "time": f"{seg.t_start:.1f}-{seg.t_end:.1f}",
                        "summary": seg.summary})
        if seg.transition_to_next:
            t = seg.transition_to_next
            changes = ", ".join(f"{k}: {v[0]}→{v[1]}" for k, v in t.entity_changes.items())
            result.append({"transition": changes})
    return result

def track(entity: str) -> list[tuple[int, EntityState]]:
    """实体状态历史 (Phase 0: 简单子串匹配)"""
    return entity_tracker.track(entity)
```

---

## 框架基座选型（已确认）

### 决策: D — 独立项目，模块化借鉴

**理由**:
- EventMemAgent / StreamAgent 均未开源 → 排除 A/C
- VITAL 非 streaming → 不适合做运行框架，但作为训练基础设施 (Phase 2/3)
- 以 proactive-project 记忆模块为基础扩展，从 StreamBridge 提取评估脚本

### VITAL 的角色: 训练基础设施 (Phase 2/3 使用)
```
Phase 2 SFT:
  - 使用 verl 框架的 fsdp_sft_trainer
  - 参考 sft_dataset_multi_turn.py 构造工具调用训练数据
  - XML tag 格式: <think>...<tool_call>...<answer>...

Phase 3 RL:
  - 使用 DGRPO 算法 (verl 已实现)
  - 难度感知: 困难问题 (跨段回忆) 获得 2x 梯度权重
  - Reward: format_reward + accuracy_reward + timing_reward
```

### 项目结构
```
streaming-agent/
├── src/
│   ├── memory/                  ← 基于 proactive-project 改造
│   │   ├── segment_store.py     ← Segment + Transition 存储与检索
│   │   ├── entity_tracker.py    ← 简单版实体追踪
│   │   ├── task_context.py      ← Layer 0 任务上下文
│   │   └── visual_buffer.py     ← Layer 2 近期帧缓冲
│   │
│   ├── perception/              ← CLIP Δ + Tier 控制
│   │   ├── clip_delta.py        ← 在线 CLIP Δ 计算
│   │   ├── keyframe_selector.py ← 段内关键帧选取
│   │   └── tier_controller.py   ← Tier 0/1/2 调度
│   │
│   ├── agent/                   ← Agent Loop + 工具
│   │   ├── agent_loop.py        ← ReAct prompt Agent (Phase 0)
│   │   ├── tools.py             ← recall/observe/diff/timeline/track
│   │   └── prompts.py           ← 工具调用提示模板
│   │
│   ├── tier1/                   ← Tier 1 段落处理
│   │   ├── summarizer.py        ← VLM 一帧生成 summary + entities
│   │   ├── transition.py        ← 纯计算段间转换
│   │   └── proactive_trigger.py ← Proactive 触发条件评估
│   │
│   ├── streaming/               ← 流式 pipeline
│   │   ├── stream_simulator.py  ← 离线视频模拟流式输入
│   │   └── pipeline.py          ← 端到端: 帧→Tier0→Tier1→Tier2→回答
│   │
│   └── eval/                    ← 评估
│       ├── ovo_bench.py         ← 从 StreamBridge 移植
│       ├── proactive_metrics.py ← 从 proactive-project 复用
│       └── error_taxonomy.py    ← A/B/C/D/E 错误分类
│
├── configs/
│   ├── default.yaml             ← 默认配置 (阈值, 模型路径等)
│   └── tools.yaml               ← 工具 schema 定义
│
├── data/                        ← 软链接到 /home/v-tangxin/GUI/data/
└── experiments/                 ← 实验结果
```

---

## 数据与 Benchmark 策略

### 已有数据匹配分析

| 数据集 | 本地路径 | 大小 | 有 QA? | 测记忆? | 测 Proactive? | 适合阶段 |
|--------|---------|------|--------|---------|--------------|---------|
| EGTEA Gaze+ | `/data/EGTEA_Gaze_Plus` | 21GB | ❌ 有动作段 GT | ❌ | ❌ | Phase 0a (切片验证) |
| ESTP-Bench | `/data/ESTP-Bench` | 31GB | ✅ 1212 QA | ⚠️ 间接 | ✅ 核心 | Phase 3 (Proactive) |
| ProAssist | `/data/ProAssist` | 240GB | ⚠️ w2t 标注 | ❌ | ✅ 触发时机 | Phase 2 训练数据源 |
| ego-dataset | `/data/ego-dataset` | 336GB | ❌ | ❌ | ❌ | 大规模验证 |

**核心缺口**: 无数据直接测试**记忆检索** (recall 有效性)，需要 OVO-Bench。

### 需下载 Benchmark

#### OVO-Bench（最高优先级）
- 来源: https://github.com/JoeLeelyf/OVO-Bench / HuggingFace
- 规模: 644 视频, 1640 QA (Backward 631 + Realtime 738 + Forward 172)
- 数据量: 标注 <1MB, 预处理切片视频 144GB, 原始视频 44GB
- 评估代码: StreamBridge 已有完整实现 (`temp/ml-streambridge/eval/eval_benchmarks/parallel_ovo_bench.py`)
- 视频来源含 Ego4D → 部分视频可能与本地 ego-dataset 重叠
- **分阶段下载策略**:
  - 立即: 下载标注文件 ovo_bench_new.json (<1MB)
  - Phase 0b: 只下载 Backward Tracing 子集对应的 20-30 个视频 (~3-5GB)
  - Phase 1: 下载完整切片视频 (144GB)

#### StreamingBench（第二优先级）
- 来源: https://github.com/THUNLP-MT/StreamingBench
- 规模: 900 视频, 4500 QA, 18 任务
- Phase 1 正式评估时下载

#### ProactiveVideoQA（Phase 3 扩展）
- 来源: arXiv:2507.09313
- Phase 3 RL 训练后才需要

### 各 Phase 数据使用规划

| Phase | 数据 | 规模 | 用途 |
|-------|------|------|------|
| 0a | EGTEA Gaze+ (已有) | 21GB | CLIP Δ 切片 vs GT 动作边界 |
| 0b | **OVO-Bench mini** (Backward 子集) | ~3-5GB (需下载) | recall/observe/diff 有效性验证 |
| 0c | OVO-Bench mini (3 类各 10 题) | ~5GB | 端到端流式 pipeline 回归测试 |
| 1 | OVO-Bench 全量 + StreamingBench | ~150GB+需下载 | 正式跑分 + Error Taxonomy |
| 2 | OVO-Bench + Ego4D NLQ 反推 | 从 Phase 1 数据构造 | SFT 训练数据 (~5K-10K 条) |
| 3 | ESTP-Bench (已有) + ProactiveVideoQA | 31GB+需下载 | Proactive timing RL |

### OVO-Bench mini 开发子集设计
从 Backward Tracing 三个子任务中各选 7-10 题:
```
EPM (情景记忆, 297题): 选 10 题 — 测 recall 基础检索能力
ASI (动作序列推理, 148题): 选 10 题 — 测 timeline + recall 组合
HLD (线索检测, 186题): 选 10 题 — 测 observe + recall 组合
```
再从 Realtime (10题) + Forward (10题) 补充，总计 ~50 题作为快速回归集。

---

## 实施路线

### Phase 0: Training-Free Agent 搭建与验证

**目标**: 验证架构本身的价值，不涉及训练

#### 0a. CLIP Δ 切片验证（1-2 天）
- 数据: EGTEA Gaze+ (本地, 有动作段 GT 标注)
- 实验: CLIP Δ 变化驱动切片 vs 固定间隔切片
- 指标: 与 GT 动作边界的 IoU / boundary F1
- 输出: 验证"变化驱动优于时间驱动"这个核心假设
- **Gate**: boundary F1 > 固定间隔 → Pass

#### 0b. 最小 Agent 验证（3-5 天）
- 数据: **OVO-Bench mini Backward 子集** (30 题, 需提前下载标注+对应视频)
- 手动给定段落（用 GT 或 0a 的结果）
- VLM (Qwen3-VL-8B) + recall/observe/diff 三个工具
- Prompt Engineering 驱动工具调用 (ReAct 格式)
- 验证: 有结构化记忆的 Agent vs 无记忆 baseline
- **Gate**: 记忆 Agent 在 Backward Tracing 上优于无记忆 baseline

#### 0c. 端到端流式 Agent（1 周）
- 数据: **OVO-Bench mini 回归集** (50 题, Backward+Realtime+Forward)
- 接入 Tier 0 (CLIP Δ 感知) + Tier 1 (summary + entity 生成)
- 完整 pipeline: 视频流 → 切片 → 记忆存储 → 按需检索 → 工具调用 → 回答
- Proactive 触发逻辑初版
- 每次代码修改后在回归集上验证

### Phase 1: 第一轮 Benchmark 评估

**目标**: 获得 baseline 分数，定位错误类型

#### 前置: 下载完整数据
- OVO-Bench 切片视频 (144GB)
- StreamingBench 数据 (待确认大小)

#### 评估组合
| Benchmark | 规模 | 目的 | 对标 |
|-----------|------|------|------|
| **OVO-Bench** | 1640 QA | 核心: Backward Tracing 测记忆检索 | EventMemAgent 60.75, StreamAgent 49.4 |
| **StreamingBench** | 4500 QA | 通用: 18 任务广覆盖 | StreamBridge 77.04, EventMemAgent 77.00 |

#### 评估代码
- OVO-Bench: 从 StreamBridge 移植 `parallel_ovo_bench.py`，替换模型调用为 Agent pipeline
- StreamingBench: 参考官方评估脚本

#### Error Taxonomy 分析
跑分后按以下分类统计错误分布:
```
A. 感知失败 — 看到正确帧但理解错误 → 需更强 VLM
B. 检索失败 — 信息在记忆中但没检索到 → summary 质量 / recall 策略
C. 遗忘失败 — 信息没被存入记忆 → CLIP Δ 阈值 / 切片粒度
D. 推理失败 — 检索到但推理错误 → Training signal (SFT)
E. 工具误用 — 选错工具或参数 → Training signal (SFT)
```

**决策点**:
- A/B/C 主导 → 回 Phase 0 优化架构
- D/E 主导 → 进入 Phase 2 训练

### Phase 2: SFT 训练（工具调用决策）

**目标**: 教会模型正确使用工具

#### 训练基础设施: VITAL verl 框架
- SFT 训练器: `verl.trainer.fsdp_sft_trainer`
- 数据加载: `sft_dataset_multi_turn.py` (支持多轮工具调用)
- 工具格式: XML tag `<think>...<tool_call>...<answer>...`

#### 数据构造
- 来源: 从 OVO-Bench / Ego4D NLQ 标注反推工具调用序列
- 格式: `(context, question) → tool_call_sequence → answer`
- 规模: ~5K-10K 条

#### 训练配置 (参考 VITAL Stage 3)
- 基座: Qwen2.5-VL-7B 或 Qwen3-VL-8B
- 冻结视觉编码器, LoRA SFT
- lr=1e-5, batch_size=256, epochs=1

### Phase 3: DGRPO RL（整体决策优化 + Proactive）

**目标**: 优化整体决策质量 + 学习 proactive timing

#### 训练基础设施: VITAL DGRPO
- 算法: DGRPO (难度感知, 困难问题获得 2x 梯度权重)
- rollout: vLLM 多轮采样, group_size=8
- actor_lr=1e-6, KL 系数=0.01

#### Reward 设计
```python
# 参考 VITAL 的 format + iou 双奖励模式
reward = (
    format_reward(response)      # <think>/<tool_call>/<answer> 格式检查
    + accuracy_reward(response)  # QA 正确性 (替代 VITAL 的 iou)
    + timing_reward(response)    # Proactive 时机准确性 (ESTP-F1)
)
# DGRPO 自动加权: 跨段回忆题 (难) 获 2x 权重, 当前帧题 (易) 获 0.5x
```

#### 评估组合 (扩展)
| Benchmark | 目的 |
|-----------|------|
| OVO-Bench | 记忆检索 + Backward Tracing |
| StreamingBench | 通用 streaming 能力 |
| **ESTP-Bench** | Proactive timing (已有实验框架!) |
| **ProactiveVideoQA** | PAUC 时间动态评估 |

#### 目标分数
| Benchmark | Phase 1 (TF) | Phase 2 (SFT) | Phase 3 (RL) | SOTA |
|-----------|:------------:|:-------------:|:------------:|:----:|
| OVO-Bench | ~45-50 | ~55-60 | **>60.75** | 71.30 (StreamBridge) |
| StreamingBench | ~65-70 | ~72-75 | **>77.00** | 77.3 (StreamForest) |
| ESTP-Bench | — | — | **>34.7** | 34.7 (EyeWO) |

---

## 已确认决策

| 编号 | 问题 | 决策 | 理由 |
|------|------|------|------|
| D1 | 框架基座 | D: 独立项目, 模块化借鉴 | EventMemAgent/StreamAgent 未开源 |
| D2 | Tier 1 summary 生成 | 1 帧 + 一次 VLM 调用 (summary+entity) | 分开准确度高, 一帧足够 |
| D3 | 段间转换 | 纯计算 entity diff, 不需 VLM | transition 可从 entity_snapshot 推导 |
| D4 | Entity Tracker | Phase 0 简单版 (子串匹配), 等 Error Taxonomy 再决定是否强化 | 先跑再看 |
| D5 | Agent Loop 格式 | Phase 0: ReAct → Phase 2: XML tag | TF 阶段 ReAct 最佳, 训练阶段 VITAL 已验证 XML tag |
| D6 | 训练框架 | VITAL verl + DGRPO | Phase 2/3 使用, 不做运行框架 |
| D7 | 核心评估 Benchmark | OVO-Bench (记忆检索) + StreamingBench (通用) | 直接测记忆系统价值, ESTP-Bench 留 Phase 3 |
| D8 | 开发迭代数据 | OVO-Bench mini (50 题子集) | 分阶段下载, 避免等 144GB 才能开始开发 |
| D9 | 数据下载策略 | 先标注 → 再子集视频 → 最后全量 | 最快启动开发循环 |

## 待决策项

| 编号 | 问题 | 选项 | 优先级 |
|------|------|------|--------|
| D10 | Tier 1 VLM 模型 | Qwen2.5-VL-3B vs Qwen3-VL-8B | 高 — 影响 summary 质量 vs 延迟 |
| D11 | 基座 VLM (Tier 2 Agent) | Qwen3-VL-8B / GPT-4o | 中 — Phase 0 两个都试 |
| D12 | Proactive 触发条件 | entity_change 幅度 / diff 阈值 / VLM 判断 | 中 — Phase 0c 实验确定 |
| D13 | OVO-Bench 视频与本地 ego-dataset 重叠度 | 下载标注后分析 | 中 — 可节省下载量 |

---

## 本地资源

### 已有数据
| 资源 | 路径 | 大小 | 用途 |
|------|------|------|------|
| EGTEA Gaze+ | `/data/EGTEA_Gaze_Plus` | 21GB | Phase 0a 切片验证 |
| ESTP-Bench | `/data/ESTP-Bench` | 31GB | Phase 3 proactive 评估 |
| ProAssist | `/data/ProAssist` | 240GB | Phase 2 训练数据源 |
| ego-dataset | `/data/ego-dataset` | 336GB | 大规模验证, 可能与 OVO-Bench 重叠 |

### 需下载数据
| 资源 | 来源 | 大小 | 优先级 | 用途 |
|------|------|------|--------|------|
| OVO-Bench 标注 | HuggingFace | <1MB | **立即** | Phase 0b 子集选取 |
| OVO-Bench mini 视频 | HuggingFace (部分) | ~3-5GB | Phase 0b 前 | 开发迭代 |
| OVO-Bench 全量切片 | HuggingFace | 144GB | Phase 1 前 | 正式评估 |
| StreamingBench | GitHub | 待确认 | Phase 1 前 | 正式评估 |
| ProactiveVideoQA | arXiv | 待确认 | Phase 3 前 | Proactive PAUC 评估 |

### 代码资源
| 资源 | 路径 | 用途 |
|------|------|------|
| StreamBridge | `temp/ml-streambridge/` | OVO-Bench 评估脚本移植 |
| VITAL | `temp/ThinkingWithVideos/` | verl 训练框架 + DGRPO |
| Proactive 项目 | `proactive-project/` | 记忆模块 + 流式仿真 + CLIP 代码 |
| ESTP 实验框架 | `proactive-project/experiments/estp_phase3/` | Proactive 评估 pipeline |
