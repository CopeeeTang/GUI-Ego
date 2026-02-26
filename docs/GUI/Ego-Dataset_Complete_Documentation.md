# Ego-Dataset 完整数据说明文档

**版本**: v1.0
**更新日期**: 2026-01-29
**数据集路径**: `/home/v-tangxin/GUI/data/ego-dataset/`

---

## 目录

1. [数据集概览](#1-数据集概览)
2. [实验设计与任务](#2-实验设计与任务)
3. [被试信息](#3-被试信息)
4. [数据采集设备](#4-数据采集设备)
5. [数据结构](#5-数据结构)
6. [标注任务详解](#6-标注任务详解)
7. [场景选择策略](#7-场景选择策略)
8. [实验流程](#8-实验流程)
9. [数据使用指南](#9-数据使用指南)
10. [附录](#10-附录)

---

## 1. 数据集概览

### 1.1 数据集定位

**Ego-Dataset** 是一个**以用户为中心的日常生活多模态数据集**，专为智能眼镜 AI 助手研究设计。数据集采集自真实日常场景，包含：

- 📹 **第一人称视角视频** (Pupil Labs Neon 眼动仪)
- 👁️ **眼动追踪数据** (注视点、瞳孔、头部运动)
- ❤️ **生理信号** (心率、皮电、血氧、PPG、体温)
- 🎙️ **音频与语音转录**
- 📝 **多层次行为与认知标注**

### 1.2 研究目标

支持以下智能助手核心能力的研究：

| 研究方向 | 描述 |
|---------|------|
| **情绪与认知识别** | 预测用户情绪状态、记录意图、主观评价 |
| **情境感知推荐** | 主动预测用户需求，提供合适时机的建议 |
| **自然交互** | 理解多模态输入的自由表达请求 |
| **个性化记忆** | 预测用户希望记住的内容，辅助长短期记忆 |

### 1.3 数据规模

```
总被试数量: 20 人 (P1-P24，缺 P17-P19, P22)
总采集时长: ~60+ 小时
单被试时长: 1-6 小时 (平均 3 小时)
场景覆盖: 12 大类场景 × 8 种活动类型
标注任务: 6 大类标注任务
```

---

## 2. 实验设计与任务

### 2.1 任务体系概览

数据集围绕 **3 大核心维度** × **6 个标注任务** 展开：

```
┌─────────────────────────────────────────────────────────┐
│  维度 1: Emotion/Cognition (情绪与认知)                    │
├─────────────────────────────────────────────────────────┤
│  • Task 1.1 - 记录瞬间预测 (MoRP)                          │
│  • Task 1.2 - 主观评价预测                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  维度 2: Requests (用户请求)                              │
├─────────────────────────────────────────────────────────┤
│  • Task 2.1 - 用户主动请求                                │
│  • Task 2.2 - 主动推荐评估                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  维度 3: Memory (记忆)                                    │
├─────────────────────────────────────────────────────────┤
│  • Task 3.1 - 记忆可达性预测                              │
│  • Task 3.2 - 记忆代理预测                                │
└─────────────────────────────────────────────────────────┘
```

---

### 2.2 Task 1.1 - Moment-of-Record Prediction (MoRP)

**负责人**: Borislav Pavlov

#### 目标
预测用户是否处于"值得记录/喜爱/有趣"的瞬间。

#### 监督信号来源
1. **In-situ 标注** (实时采集中)
   - 用户通过**语音指令**触发记录：
     - `"Make a photo"` - 拍照
     - `"Start video"` - 开始录像
     - `"Stop video"` - 结束录像
   - 每次指令后，用户**口述理由**：
     - 例："Make a photo because I like this painting"
     - 例："Start video because I want to record this shop"

2. **Post-hoc 标注** (事后标注)
   - 语音转文字后，标注员提取：
     - 记录类型 (photo/video)
     - 理由描述 (reason description)
     - 触发类型 (reason/trigger type)：
       - Emotional reaction (情感反应)
       - Information moment (信息瞬间)
       - Visual/aesthetic curiosity (视觉/美学好奇)
       - Social moment (社交瞬间)
     - 注意力水平 (attention level)
     - 自发程度 (spontaneity level)

#### 相关传感器信号
- 眼动行为：反复注视、瞳孔变化
- 心率/HRV/PPG 等生理指标
- 环境数据：位置、运动、音频特征

#### 数据示例
```json
{
  "timestamp": 1234.56,
  "action": "make_photo",
  "reason": "I like this painting",
  "trigger_type": "visual_curiosity",
  "attention_level": "high",
  "spontaneity": "high"
}
```

---

### 2.3 Task 1.2 - Subjective Evaluation Prediction

**负责人**: 苟芳菲

#### 目标
预测用户在**回顾阶段**对经历事件的主观情绪与评价（不同于即时反应）。

#### 监督信号来源
实验结束后，用户以 **vlog 形式**回顾当天经历：
- 自主讲述印象深刻的瞬间
- 可查阅视频作为辅助（非强制完整回看）
- 筛选关键片段并组合

#### 标注内容

| 标注项 | 类型 | 说明 |
|-------|------|------|
| **时间片段** | [start, end, duration] | 被试选中的视频片段 |
| **情绪类别** | 多选 | 见下表 |
| **情绪程度** | 3/5-point Likert | 强度评分 |
| **主观描述** | 文本 | 用户口述的感受 |
| **相关物体** | Bounding box | 触发情绪的视觉对象 |

#### 情绪类别 (9 类)

| 情绪 | 中文 | 典型想法 |
|-----|------|---------|
| **Joy** | 快乐 | "I'm really happy / this is great" |
| **Surprise** | 惊讶 | "I didn't expect that / wow" |
| **Achievement** | 成就感 | "I did it / this is valuable" |
| **Social Connection** | 社交连接 | "We're together / I feel understood" |
| **Relaxation** | 放松 | "I feel calm / there's no pressure" |
| **Stress** | 压力 | "I'm nervous / under stress" |
| **Disappointment** | 失望 | "I feel let down / not so good" |
| **Boring** | 无聊 | - |
| **Awkward** | 尴尬 | - |

#### 标注数据格式
```json
{
  "start": 35,
  "end": 42,
  "duration": 7,
  "emotions": [
    {
      "emotion_label": "Achievement",
      "emotion_level": 3,
      "description": "Finally finished my assignment. I felt relieved."
    }
  ],
  "objects": [
    {
      "X": 54.3,
      "Y": 1200.9,
      "Width": 860,
      "Height": 393,
      "label": "assignment paper"
    }
  ]
}
```

---

### 2.4 Task 2.1 - User-Initiated Free-Form Requests

**负责人**: 王泽宇

#### 目标
采集用户向 AI 助手发出的自然语言请求，预测意图与参数。

#### 监督信号来源

1. **实时采集**
   - 用户通过**手势**标记问题开始/结束
   - 可使用多模态表达（眼神、手势、语音）

2. **事后标注**
   - 转录语音，提取所有请求及时间戳
   - 用户标注**期望的 AI 响应**（关注要点而非具体措辞）

#### 问题类型 (8 类)

| 类型 | 示例 |
|-----|------|
| **定义型** | "这道菜是什么？" "这个工具的学名是什么？" |
| **解释型** | "根据图纸，这一步应该怎么完成？" |
| **比较型** | "哪一款产品更符合我的需求？" |
| **主观检索型** | "今天早上我喂猫了吗？" "午餐花了多少钱？" |
| **算数型** | "帮我算一下这些商品的总费用" |
| **翻译型** | "翻译一下这个书名" |
| **客观信息型** | "这里允许停车吗？" "怎么解决这个报错？" |
| **辅助型** | "请将会议要点梳理一下" |

---

### 2.5 Task 2.2 - Proactive Request Recommendations

**负责人**: Eduardus, 徐和平, 王泽宇

#### 目标
预测用户当前是否适合接收 AI 主动推荐，并对推荐排序。

#### 工作流程

1. **事前准备**
   - 被试讲述自己可/不可被打扰的标准

2. **推荐生成**
   - 使用 naive 算法为视频片段生成候选推荐

3. **用户标注**
   - 对每个视频片段及其推荐：
     - ✅ 是否适合推荐？
     - 📊 推荐排序（可接受度）
     - 🔧 自定义更合适的请求
   - 标注维度：
     - 重要性 (importance)
     - 紧急性 (urgency)
     - 关联性 (relevance)

#### 推荐意图类型 (9 类)

1. **Context-aware recommendation** - 情境感知推荐
2. **Task assistance** - 任务辅助
3. **Decision support** - 决策支持
4. **Object identification** - 物体识别
5. **Procedural guidance** - 流程引导
6. **Computation & estimation** - 计算估算
7. **Contextual memory** - 情境记忆
8. **Translation & recognition** - 翻译识别
9. **Engaging interaction** - 趣味互动

#### 标注示例
```json
{
  "timestamp_start": 1000,
  "timestamp_end": 1100,
  "is_good_time_for_recommendation": true,
  "recommendations": [
    {
      "rank": 1,
      "intent": "task_assistance",
      "content": "需要我帮你记录这个商品信息吗？",
      "importance": 4,
      "urgency": 2,
      "relevance": 5
    }
  ]
}
```

---

### 2.6 Task 3.1 - Memory Reachability Prediction

**负责人**: 刘畅

#### 目标
预测用户在**延迟回忆**中：
- 某段经历是否能被回忆起（记忆成功/遗忘）
- 回忆内容的准确度
- 回忆叙述的细节丰富度

#### 工作流程

1. **事件自动划分**
   - 使用时序分割算法：
     - 行为变换点检测
     - 场景切换检测
     - 对象交互信号
   - 输出：连续、语义一致的事件片段 + 时间戳

2. **延迟回忆环节** (7-14 天后)
   - 用户口述回忆每个事件
   - 标记：Reachable (1) / Unreachable (0)

#### 回忆模板

```yaml
总体:
  recall_confidence: 高/中/低

事件 (Event):
  location: 地点
  participants: 参与者
  main_activities: 主要活动
  key_objects_actions: 关键物品/动作

关键细节 (Key Details):
  vivid_details: [生动的细节1, 细节2, ...]
  vague_events: [模糊的地方]
  why_cannot_remember: 为什么想不起来了

情绪:
  emotion: 当时的情绪
```

#### 注意事项
- 可先标注其他任务（0-2 天内）
- Task 3.1 隔 **7-14 天**再标注（测试长期记忆）

---

### 2.7 Task 3.2 - Memory Agent Prediction

**负责人**: 刘畅

#### 目标
预测用户希望 AI **记忆代理**记住的内容（区分有用信息 vs 隐私/无用信息）。

#### 标注内容

用户观看视频后：

| 标注项 | 说明 |
|-------|------|
| **要存储的片段** | 绘制视频时间段 |
| **关键物品/场景** | Bounding box 标注 |
| **标记原因** | 文本或语音解释 |

#### 标注示例
```json
{
  "segment": {
    "start": 500,
    "end": 520,
    "reason": "停车位置，下次要找车"
  },
  "key_objects": [
    {
      "bbox": [100, 200, 300, 400],
      "label": "Parking lot sign B2-14",
      "reason": "记住停车位编号"
    }
  ]
}
```

---

## 3. 被试信息

### 3.1 基本统计

- **总人数**: 20 人 (实际招募 32 人，数据质量筛选后保留 20 人)
- **编号**: P1-P24（缺失 P17-P19, P22）
- **年龄范围**: 21-30 岁
- **性别**: 男女混合
- **背景**: 主要为清华大学学生/研究人员

### 3.2 被试元数据

每位被试包含以下信息：

| 字段 | 示例 |
|-----|------|
| **Collection_ID** | C001, C002, ... |
| **Participant_ID** | P001, P002, ... |
| **姓名** | Yue Pan, Eduardus Tjitrahardja |
| **年龄** | 22 |
| **性别** | Female / Male |
| **用户画像** | 详细个人档案（见下） |
| **采集日期** | 2025-12-21 |
| **标注日期** | 2025-12-21 |
| **Task 3.1 回忆间隔** | 1h / 1 week / 1 month |

### 3.3 用户画像示例 (User Profile)

用于 Task 2.2 个性化推荐生成：

```yaml
Name / Preferred Name: Sarah
Identity / Occupation: 清华大学管理科学与工程专业学生
Language Abilities: 中文母语；英语流利
Personality Traits: 对自然和科技功能充满好奇，喜欢与外国人交流
Interests / Hobbies: 阅读物理和社会学书籍，钢琴演奏
Recent Goals / Plans: 完成第一篇学术论文
```

### 3.4 场景上下文 (Scene Context)

部分被试包含多段采集，每段有独立场景描述：

**P002 (Eduardus) 的场景示例**：

```yaml
- start_time: 0
  end_time: 2397
  scene_context: >
    准备下周期末考试的 Eduardus 与同事乘出租车从清华大学
    前往北京 798 艺术区参观艺术博物馆。

- start_time: 2397
  end_time: 5149
  scene_context: >
    Eduardus 抵达 798 艺术区并开始与同事探索艺术博物馆。
    这是他第一次参观该区域，对周围环境不熟悉。
```

---

## 4. 数据采集设备

### 4.1 设备清单

| 设备类型 | 型号 | 数量 | 用途 |
|---------|------|------|------|
| **眼动仪眼镜** | Pupil Labs Neon | 1 | 第一视角视频 + 眼动 + IMU |
| **智能手表** | 生理信号手表 | 1 | 心率/EDA/PPG/血氧/温度 |
| **智能戒指** | Ring | 1 | 补充生理信号 |
| **配对手机** | OnePlus / iPhone | 1 | 数据同步与控制 |
| **Apple Watch** | (部分被试) | 1 | 额外心率数据 |

### 4.2 设备对齐 (Cross-Device Alignment)

由于多设备独立运行，需进行时间戳对齐：

#### 对齐方法
1. **拍手对齐** (Clap Hand)
   - 开始时拍手 → 记录各设备时间戳
   - 结束时拍手 → 验证时间漂移

2. **PPG 对齐**
   - 利用手表 PPG 信号的心跳峰值对齐

3. **屏幕闪烁对齐** (部分被试)
   - 手表屏幕闪烁 → 眼镜视频捕捉

#### 对齐记录示例 (P001)
```yaml
Watch start time: 10:33:00
Ring start time: 10:32:00
Glasses start time: 10:32:00
Start Align - clap hand: 10:34:00
Start Align - PPG: 10:34:00
```

---

## 5. 数据结构

### 5.1 整体目录结构

```
/home/v-tangxin/GUI/data/ego-dataset/
├── data/                          # 主数据集
│   ├── P1_YuePan/
│   ├── P2_EduardusTjitrahardja/
│   ├── P3_BorislavPavlov/
│   ├── ...
│   └── P24_ChenYiyi/
├── test_data/                     # 测试数据
│   ├── data_download_from_pupilcloud/
│   └── raw_data_export_from_phone/
├── transript/                     # 独立语音转录
│   ├── audio-yue.wav
│   └── audio.wav
└── README.md
```

### 5.2 单个被试文件夹结构

以 **P1_YuePan** 为标准结构：

```
P1_YuePan/
├── raw_data/                      # 原始传感器数据
│   ├── 2025-12-21_10-32-27/      # Pupil Labs Neon 数据
│   │   ├── gaze.csv              # 眼动数据 (200 Hz)
│   │   ├── imu.csv               # IMU 数据 (92 Hz)
│   │   ├── *.mp4                 # 场景视频 (30 fps, 960x720)
│   │   ├── audio.wav             # 原始音频
│   │   ├── info.json             # 元数据
│   │   └── ...
│   └── Watch_10_2025-12-21_10-33-00/  # 手表数据
│       ├── acc.csv               # 加速度+陀螺仪 (20 Hz)
│       ├── hr.csv                # 心率 (1 Hz)
│       ├── eda.csv               # 皮电 (10 Hz)
│       ├── ppg.csv               # 光电容积 (100 Hz)
│       ├── o2.csv                # 血氧
│       └── skt.csv               # 皮肤温度
│
├── annotation/                    # 标注文件
│   ├── annotations_1.1_*.json    # Task 1.1 - 记录瞬间
│   ├── annotations_1.2_*.json    # Task 1.2 - 情绪标注
│   ├── annotations_2.1_*.json    # Task 2.1 - 用户请求
│   ├── annotations_2.2_*.json    # Task 2.2 - 推荐评估
│   ├── annotations_3.2_*.json    # Task 3.2 - 记忆代理
│   ├── forbidden_segments_*.json # 免打扰时段
│   └── transcripts/              # 语音转录
│       └── *.json
│
└── Timeseries Data + Scene Video/  # (部分被试有)
    └── merged_timeseries.csv     # 合并的时序数据
```

### 5.3 原始数据文件详解

#### 5.3.1 Pupil Labs Neon 眼动仪

**文件**: `raw_data/YYYY-MM-DD_HH-MM-SS/`

| 文件 | 采样率 | 字段 | 单位 |
|-----|--------|------|------|
| **gaze.csv** | 200 Hz | `gaze x [px]`, `gaze y [px]` | 像素 |
|  |  | `azimuth [deg]`, `elevation [deg]` | 度 |
|  |  | `fixation id` | - |
| **imu.csv** | ~92 Hz | `gyro x/y/z [deg/s]` | 度/秒 |
|  |  | `acceleration x/y/z [m/s^2]` | m/s² |
|  |  | `quaternion w/x/y/z` | - |
|  |  | `roll/pitch/yaw [deg]` | 度 |
| **world.mp4** | ~30 fps | 场景视频 | 960×720 |
| **audio.wav** | - | 原始音频 | - |

**注意**: 视频通常分段存储（每段约 1-2 小时）

#### 5.3.2 智能手表生理信号

**文件**: `raw_data/Watch_[ID]_[Date]_[Time]/`

| 文件 | 采样率 | 字段 | 单位 |
|-----|--------|------|------|
| **acc.csv** | 20 Hz | `acc_x/y/z` | m/s² |
|  |  | `gyro_x/y/z` | rad/s |
| **hr.csv** | 1 Hz | `heart_rate` | bpm |
| **eda.csv** | 10 Hz | `eda` | μS |
| **ppg.csv** | 100 Hz | `ppg_green/red/ir` | - |
| **o2.csv** | - | `spo2` | % |
| **skt.csv** | - | `skin_temperature` | °C |

### 5.4 标注数据文件详解

所有标注文件均为 **JSON 格式**，存储在 `annotation/` 文件夹。

#### 命名规范
```
annotations_[Task]_[SubjectID]_[Date].json
```

示例：
- `annotations_1.1_P001_2025-12-21.json`
- `annotations_2.2_P002_recommendation_candidates.json`

#### 特殊文件
- `forbidden_segments_*.json` - 隐私保护时段
- `transcripts/*.json` - 语音转文字结果

---

## 6. 标注任务详解

### 6.1 标注文件对应关系

| 任务 | 文件名模式 | 标注时机 | 回忆间隔 |
|-----|-----------|---------|---------|
| Task 1.1 | `annotations_1.1_*.json` | 录制中 + 事后 | - |
| Task 1.2 | `annotations_1.2_*.json` | 事后 | 当天 |
| Task 2.1 | `annotations_2.1_*.json` | 录制中 + 事后 | - |
| Task 2.2 | `annotations_2.2_*.json` | 事后 | 当天 |
| Task 3.1 | (口述录音) | 事后 | 7-14 天 |
| Task 3.2 | `annotations_3.2_*.json` | 事后 | 当天 |

### 6.2 数据规模参考 (P1_YuePan)

| 指标 | 数量/规模 |
|-----|----------|
| 录制时长 | 3.17 小时 (11,223 秒) |
| 眼动数据点 | ~2,240,000 |
| 视觉目标标注 (Task 1.1) | 14 个 |
| 情绪标注 (Task 1.2) | 15 段 |
| 语音问答 (Task 2.1) | 19 条 |
| AI 推荐候选 (Task 2.2) | 330+ 条 |
| 用户接受推荐 | 40+ 条 (~12% 接受率) |
| 隐私屏蔽时段 | 25 段 |

---

## 7. 场景选择策略

### 7.1 场景选择原则

根据官方文档，场景选择遵循以下原则：

✅ **包含**:
- 日常生活中常见场景
- 活动丰富且个性化
- 允许设备录制的公共/私人空间

❌ **排除**:
- 高强度运动（眼镜佩戴限制）
- 涉及隐私的场景（卫生间）
- 大段无互动的学习场景

### 7.2 场景分类 (12 大类)

| 场景类型 | 具体环境 | 典型活动 |
|---------|---------|---------|
| **1. 个人居所** | 家、宿舍、厨房、阳台、客厅 | 做饭、打扫、整理桌面、网购 |
| **2. 工作学习** | 教室、会议室、图书馆、自习室 | 记笔记、小组讨论、查资料 |
| **3. 交通/寻路** | 地铁、公交、街道、停车场、机场 | 换乘、导航、寻找停车位 |
| **4. 运动健身** | 健身房、球场、操场 | 轻度运动（不佩戴眼镜） |
| **5. 饮食场所** | 食堂、餐厅、咖啡馆、奶茶店 | 点餐、品尝、等位、结账 |
| **6. 娱乐场所** | 手工作坊、电玩城、棋牌室、轰趴馆 | 玩桌游、手工、K歌 |
| **7. 购物场所** | 商店、超市、菜市场、商场 | 挑选商品、比价、查看成分 |
| **8. 旅游景点** | 人文景点、自然景观 | 观光、拍照、导览 |
| **9. 特殊活动** | 书市、画展、演出、集市 | 参观、互动、欣赏 |
| **10. 服务办理** | 快递站、柜台、收银台、机场值机 | 办理业务、排队、咨询 |
| **11. 户外环境** | 公园、绿地、街区、河边、海滩 | 散步、徒步、观鸟、露营 |
| **12. 社交环境** | 聚会、包间、活动中心 | 聊天、游戏、合影 |

### 7.3 活动类型 (8 大类)

| 活动类型 | 描述 | 示例 | 相关任务 |
|---------|------|------|---------|
| **任务型** | 分步骤完成的任务 | 做饭、手工、搭建家具 | Task 3.2 |
| **娱乐型** | 广义娱乐活动 | 运动、购物、玩桌游、唱歌 | Task 1.1, 1.2 |
| **学习型** | 学习相关活动 | 整理笔记、读书、写作业 | Task 2.1 |
| **交互型** | 多人交互 | 小组讨论、服务办理、开会 | Task 2.1, 2.2 |
| **日常型** | 日常琐事 | 导航、刷视频、购物、打电话 | All tasks |
| **记录型** | 需日后查找的记录 | 风景、抢票信息、备忘录 | Task 1.1, 3.2 |
| **信息提取型** | 从资料获取信息 | 货架标签、指路牌、说明书 | Task 2.1 |
| **比较型** | 对比选择 | 商品比较、攻略比较 | Task 2.1, 2.2 |

### 7.4 场景捕捉方式

| 捕捉类型 | 用途 | 示例 |
|---------|------|------|
| **单一信息静止图** | 定义/检索/解释问题 | 标签、信息面板 |
| **比较信息对比图** | 比较类问题 | 多商品并排对比 |
| **事前/事后状态图** | 展示变化 | 收纳前后、整理前后 |
| **记忆锚点图** | 记忆提示 | 停车位、网站域名 |
| **地图导览图** | 导航 | 导航软件、商场平面图 |
| **收据账单图** | 算数/检索问题 | 账单、发票 |
| **操作视频** | 任务流程 | 做饭过程、组装家具 |
| **活动视频** | 活动记录 | 小组讨论、比赛 |

### 7.5 完整场景示例

#### 示例 1: 居家环境

```yaml
环境: 厨房、阳台、卧室、书桌、客厅
活动: 做饭、晾衣服、打扫卫生、整理桌面、网上购物
捕捉场景:
  - 活动工序、工具和原料
  - 环境中的其他物品
  - 活动前后的环境差异
  - 购物车商品详情图
相关问题:
  - "白色的牛仔裤洗了吗？"
  - "肥牛卷应该冷水下锅吗？"
  - "收拾桌面的时候把颜料盒放到哪了？"
  - "这个污渍擦不掉怎么办？"
  - "这两种耳塞哪个性价比更高？"
```

#### 示例 2: 购物环境

```yaml
环境: 超市、商场、便利店、菜市场
活动: 挑选商品、比价、查看成分/材质、寻找店铺、排队结账
捕捉场景:
  - 商品货架陈列
  - 价格标签、成分表
  - 店铺导览图
  - 促销活动海报
相关问题:
  - "这款洗发水和那款在成分上有什么区别？"
  - "这个品牌的酸奶今天有折扣吗？"
  - "儿童玩具区在几楼？"
  - "这件衣服有M码吗？"
  - "哪个品牌的空气炸锅口碑更好？"
```

---

## 8. 实验流程

### 8.1 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: 实验前准备 (1-2 天前)                                 │
├─────────────────────────────────────────────────────────────┤
│ 1. 被试规划场景与活动                                          │
│ 2. 与实验者确认计划                                            │
│ 3. 设备使用方法培训                                            │
│ 4. 获取录制许可 (地点 + 人物)                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: 数据采集 (3-6 小时)                                   │
├─────────────────────────────────────────────────────────────┤
│ 1. 启动并调整设备 (对齐时间戳)                                 │
│ 2. 拍手对齐 (开始)                                             │
│ 3. 连续录制 (至少 1-2h 有效视频)                               │
│    - 覆盖 3-4 个场景                                           │
│    - In-situ 标注 (语音指令 + 手势)                            │
│    - 口述讲解补充                                              │
│ 4. 拍手对齐 (结束)                                             │
│ 5. 归还设备，导出数据                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: 标注 (当天或 0-2 天内)                                │
├─────────────────────────────────────────────────────────────┤
│ Task 1.2 - 情绪标注 (vlog 形式)                                │
│ Task 2.1 - 用户请求标注                                        │
│ Task 2.2 - 推荐评估                                            │
│ Task 3.2 - 记忆代理标注                                        │
│ 隐私脱敏处理                                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: 延迟回忆 (7-14 天后)                                  │
├─────────────────────────────────────────────────────────────┤
│ Task 3.1 - 记忆可达性测试 (口述回忆)                           │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 数据采集要求

#### 时长要求
- **连续时间跨度**: ≥6 小时
- **有效视频数据**: ≥1-2 小时
- **场景覆盖**: ≥3-4 个场景

#### 质量要求
- ✅ 光线充足（室外选择亮度清晰时段）
- ✅ 至少一段移动/走路的平稳捕捉
- ✅ 手和物品可见（便于标注）
- ✅ 尽量减少人脸/屏幕出现（隐私保护）
- ✅ 多人场景尽可能多互动

#### 设备检查清单
- [ ] Pupil Labs Neon 正常录制
- [ ] 手表/戒指正常采集
- [ ] 时间戳对齐完成（拍手）
- [ ] 音频清晰可用

---

## 9. 数据使用指南

### 9.1 适用研究方向

| 研究方向 | 推荐任务 | 核心特点 |
|---------|---------|---------|
| **可穿戴 AI 助手** | All tasks | 真实场景、长时跨度、多模态 |
| **情境感知推荐系统** | Task 2.2 | 330+ 候选推荐 + 用户反馈 |
| **情绪识别** | Task 1.2 | 9 类情绪 + 生理信号对齐 |
| **长视频理解** | Task 1.1, 3.1 | 3-6h 连续视频 + 事件分割 |
| **视觉问答 (VQA)** | Task 2.1 | 8 类问题 + 第一视角视频 |
| **记忆辅助系统** | Task 3.1, 3.2 | 延迟回忆 + 记忆意图预测 |
| **Generative UI** | Task 2.2 | 情境感知界面自动生成 |
| **眼动行为分析** | All tasks | 200 Hz 眼动 + 瞳孔数据 |
| **生理计算** | Task 1.1, 1.2 | 心率/EDA/PPG 同步 |

### 9.2 数据加载示例

#### 9.2.1 加载眼动数据

```python
import pandas as pd
import numpy as np

# 加载注视点数据
gaze_df = pd.read_csv('P1_YuePan/raw_data/2025-12-21_10-32-27/gaze.csv')

# 主要字段
print(gaze_df.columns)
# ['timestamp [ns]', 'gaze x [px]', 'gaze y [px]',
#  'azimuth [deg]', 'elevation [deg]', 'fixation id']

# 采样率检查
time_diff = gaze_df['timestamp [ns]'].diff().median() / 1e9
print(f"Sampling rate: {1/time_diff:.2f} Hz")  # ~200 Hz
```

#### 9.2.2 加载生理信号

```python
# 加载心率数据
hr_df = pd.read_csv('P1_YuePan/raw_data/Watch_10_2025-12-21_10-33-00/hr.csv')

# 加载皮电数据
eda_df = pd.read_csv('P1_YuePan/raw_data/Watch_10_2025-12-21_10-33-00/eda.csv')

# 加载 PPG 数据
ppg_df = pd.read_csv('P1_YuePan/raw_data/Watch_10_2025-12-21_10-33-00/ppg.csv')
```

#### 9.2.3 加载标注数据

```python
import json

# 加载情绪标注 (Task 1.2)
with open('P1_YuePan/annotation/annotations_1.2_P001.json', 'r') as f:
    emotion_annotations = json.load(f)

# 结构示例
for annotation in emotion_annotations:
    print(f"Time: {annotation['start']}-{annotation['end']}")
    print(f"Emotions: {annotation['emotions']}")
    print(f"Description: {annotation['emotions'][0]['description']}")
```

### 9.3 时间对齐策略

由于多设备独立运行，需进行时间戳对齐：

```python
def align_timestamps(metadata_file):
    """
    根据 metadata 中的对齐记录同步时间戳
    """
    # 读取对齐记录
    with open(metadata_file) as f:
        align_data = json.load(f)

    # 提取拍手对齐时间
    watch_start = align_data['watch_start_time']
    glasses_start = align_data['glasses_start_time']
    clap_time_watch = align_data['start_align_clap_watch']
    clap_time_glasses = align_data['start_align_clap_glasses']

    # 计算时间偏移
    offset = clap_time_glasses - clap_time_watch

    return offset
```

### 9.4 隐私保护

数据集已进行隐私脱敏处理：

✅ **已处理**:
- 人脸模糊化
- 敏感屏幕信息模糊
- 文字信息模糊（车牌、身份证等）

⚠️ **使用注意**:
- 仅限学术研究使用
- 不得用于商业目的
- 不得尝试反向识别被试身份

---

## 10. 附录

### 10.1 数据集统计总览

| 维度 | 统计 |
|-----|------|
| **总被试数** | 20 人 |
| **总录制时长** | ~60+ 小时 |
| **平均单被试时长** | 3 小时 |
| **视频总量** | ~40+ 段 |
| **场景覆盖** | 12 大类场景 |
| **活动类型** | 8 大类活动 |
| **眼动数据点** | ~40M+ |
| **标注任务** | 6 类 |
| **推荐候选总数** | ~6000+ 条 |

### 10.2 文件命名规范总结

#### 原始数据
```
raw_data/
├── YYYY-MM-DD_HH-MM-SS/           # Pupil Labs Neon
└── Watch_[ID]_YYYY-MM-DD_HH-MM-SS/ # 智能手表
```

#### 标注数据
```
annotation/
├── annotations_[Task]_[PID]_[Date].json
├── forbidden_segments_[PID].json
└── transcripts/[PID]_[Date].json
```

### 10.3 数据完整性检查清单

使用前请检查：

```bash
# 1. 检查必需文件夹
- [ ] raw_data/ 存在
- [ ] annotation/ 存在

# 2. 检查眼动数据
- [ ] gaze.csv 存在且非空
- [ ] world.mp4 存在且可播放
- [ ] imu.csv 存在

# 3. 检查生理数据
- [ ] Watch_* 文件夹存在
- [ ] hr.csv, eda.csv, ppg.csv 存在

# 4. 检查标注数据
- [ ] 至少有 3 个标注文件 (Task 1.1, 1.2, 2.1)
- [ ] JSON 格式正确
```

### 10.4 常见问题 (FAQ)

#### Q1: 为什么部分被试缺少 Timeseries 文件夹？
**A**: Timeseries 是合并后的预处理数据，部分被试尚未生成。可使用 `raw_data/` 中的原始数据。

#### Q2: 如何处理视频分段问题？
**A**: Pupil Labs Neon 自动分段存储视频。使用时需按时间戳排序合并：
```python
import glob
videos = sorted(glob.glob('raw_data/*/world*.mp4'))
```

#### Q3: Task 3.1 为什么没有 JSON 标注文件？
**A**: Task 3.1 (记忆可达性) 采用口述回忆方式，数据为音频录音 + 转录文本。

#### Q4: 不同被试的标注格式是否一致？
**A**: 早期被试（P1-P5）可能存在微小格式差异。建议使用 P10 之后的数据，格式更加规范。

#### Q5: 如何引用本数据集？
**A**:
```bibtex
@dataset{ego_dataset_2025,
  title={Ego-Dataset: A User-Centered Daily Life Dataset for Wearable AI},
  author={Tsinghua University},
  year={2025},
  note={Multi-modal egocentric dataset with 20 participants}
}
```

### 10.5 数据结构差异记录

| 被试 | raw_data 命名 | annotation 命名 | Timeseries | 特殊说明 |
|-----|--------------|----------------|-----------|---------|
| P1_YuePan | `raw_data/` | `annotation/` | ✅ | 有额外 Timeseries 文件夹 |
| P2_Eduardus | `raw_data/` | `annotation/` | ❌ | 3 个不同时间段采集 |
| P3_Borislav | `raw_data/` | `annotation/` | ✅ | (已规范化) |
| P10_Ernesto | `raw_data/` | `annotation/` | ❌ | transcripts → transcript (单数) |
| P16_XiangZeng | `raw_data/` | `annotation/` | ❌ | (临时文件夹已清理) |
| P20_WilliamHu | `raw_data/` | `annotation/` | ❌ | 缺少 task1.2, 3.2 标注 |
| P24_ChenYiyi | `raw_data/` | `annotation/` | ❌ | (已规范化) |

**注**: 已于 2026-01-29 完成数据结构规范化，所有被试现已统一命名。

---

## 联系方式

- **项目负责人**: 王泽宇
- **标注协调**: Eduardus Tjitrahardja, Borislav Pavlov, 刘畅, 苟芳菲
- **数据托管**: 清华大学自强科技楼

---

**文档版本历史**:
- v1.0 (2026-01-29) - 初始版本，基于官方文档与数据分析生成
