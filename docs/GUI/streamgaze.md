# StreamGaze Benchmark 深度技术分析报告

> **论文**: StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos
> **作者**: Daeun Lee et al. (UNC Chapel Hill & Adobe Research)
> **仓库**: https://github.com/daeunni/StreamGaze
> **分析日期**: 2026-02-09

---

## 1. 概述

StreamGaze 是首个专门评估多模态大语言模型（MLLMs）如何利用人类眼动信号进行流式视频时序推理和主动理解的 benchmark。

### 1.1 核心统计

| 指标 | 数值 |
|------|------|
| 视频总数 | 285 |
| QA 对总数 | 8,521 |
| 任务类别 | 10 种 (4 Past + 4 Present + 2 Proactive) |
| 支持数据集 | EGTEA-Gaze+, Ego4D-Gaze, HoloAssist, EgoExoLearn |

### 1.2 核心创新点

1. **眼动引导的时序推理**: 首次将人类注视点作为显式信号融入视频理解任务
2. **流式视频设置**: 模型必须处理时序逐帧输入，模拟 AR 眼镜等可穿戴设备场景
3. **主动式任务设计**: 包含需要模型预测未来并主动提醒的任务类型

---

## 2. 视频预处理 Pipeline

StreamGaze 提供了完整的端到端数据生成流水线，位于 `pipeline/` 目录。

### 2.1 Pipeline 架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    StreamGaze Data Generation Pipeline                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Step 0: Gaze Projection (HoloAssist only)                              │
│     └── 3D 眼动坐标 → 2D 图像平面投影                                    │
│                                                                          │
│  Step 1: Fixation Extraction (I-VT Algorithm)                           │
│     └── 原始眼动数据 → 注视点分割                                        │
│                                                                          │
│  Step 1.5: Quality Filtering                                            │
│     └── 注视点过滤与合并                                                 │
│                                                                          │
│  Step 2: Object Identification (InternVL-3.5 38B)                       │
│     └── 注视区域 → 物体识别与场景描述                                    │
│                                                                          │
│  Step 2.5: Sequence Filtering                                           │
│     └── 序列过滤与元数据合并                                             │
│                                                                          │
│  Step 3: QA Generation                                                  │
│     └── 生成 Past/Present/Future 任务的 QA 对                           │
│                                                                          │
│  Step 4: QA Validation (Qwen3VL 30B)                                    │
│     └── QA 质量验证与过滤                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Step 0: 3D 眼动投影 (HoloAssist 专用)

**文件**: `pipeline/step0_gaze_projection.py`

HoloAssist 数据集使用 HoloLens 设备采集，眼动数据为 3D 世界坐标，需要投影到 2D 图像平面。

```python
# 核心投影流程
def project_gaze_point_to_camera(origin_world, dir_world, cam_pose_world, eye_dist):
    """
    1. 眼动射线: P(t) = origin + t * dir, t = eye_dist (米)
    2. 世界坐标 → 相机坐标: inv(cam_pose)
    3. 坐标系对齐: AXIS_TRANSFORM (HoloLens → OpenCV)
    4. 针孔投影: u = fx*X/Z + cx, v = fy*Y/Z + cy
    """
    point_world = origin_world + dir_world * eye_dist
    cam_from_world = np.linalg.inv(cam_pose_world)
    Pc = AXIS_TRANSFORM @ (cam_from_world @ Pw)
    return Pc[:3] if Pc[2] > 0 else None
```

**输入数据结构**:
```
Export_py/
├── Video_pitchshift.mp4      # RGB 视频
├── Video/
│   ├── Intrinsics.txt        # 相机内参 (fx, fy, cx, cy, width, height)
│   ├── Pose_sync.txt         # 相机位姿 (时间戳, 4x4 变换矩阵)
│   └── VideoMp4Timing.txt    # 视频时间戳 (100ns 单位)
└── Eyes/
    └── Eyes_sync.txt         # 眼动数据 (origin_xyz, direction_xyz, valid)
```

**输出**: 每帧的 2D 注视坐标 CSV 文件 `{session}_gaze_2d.csv`

### 2.3 Step 1: 注视点提取 (I-VT 算法)

**文件**: `pipeline/step1_extract_fixation.py`, `pipeline/preprocess/gaze_processing.py`

使用 **I-VT (Velocity-Threshold Identification)** 算法从原始眼动数据中提取注视点 (Fixation)。

```python
def extract_fixation_segments(df, radius_thresh=0.05, duration_thresh=0.5, gap_thresh=0.2):
    """
    基于速度阈值的注视点提取

    参数:
        radius_thresh: 最大允许偏移距离 (归一化坐标)
        duration_thresh: 最小注视持续时间 (秒)
        gap_thresh: 允许的短暂中断时间 (秒)

    返回:
        fixations: [{start_time, end_time, center_x, center_y, duration}, ...]
    """
```

**眼动类型分类**:
| 类型 | 值 | 含义 |
|------|-----|------|
| Untracked | 0 | 无有效数据 |
| Fixation | 1 | 注视 (视线停留) |
| Saccade | 2 | 扫视 (视线跳跃) |
| Unknown | 3 | 未知类型 |
| Truncated | 4 | 超出视野范围 |

**多数据集支持**:
```python
# EGTEA-Gaze+: BeGaze 格式 TXT
def parse_gtea_gaze(filename, gaze_resolution=[960, 1280])

# Ego4D-Gaze: CSV 格式 (norm_pos_x, norm_pos_y, confidence)
def parse_ego4d_gaze(csv_path, fps=30)

# HoloAssist: Step 0 输出的 2D 坐标
def parse_holoassist_gaze(csv_path, fps=24.46, video_resolution=(896, 504))

# EgoExoLearn: NPY 格式 [x_norm, y_norm, validity]
def parse_egoexo_gaze(npy_path, fps=30)
```

### 2.4 Step 2: 物体识别 (InternVL-3.5 38B)

**文件**: `pipeline/step2_egtea_gaze_object_internvl.py`, `pipeline/preprocess/internvl_processor.py`

使用 **InternVL-3.5 38B** 视觉语言模型识别注视区域内的物体。

**核心流程**:

```python
def extract_objects_and_scene_from_video_clip_internvl_v2_sequential(
    requests_data,      # 注视点列表
    fov_radius,         # 视野半径 (像素)
    object_pool=None,   # 已识别物体池
    temperature=0.3
):
    """
    两阶段物体识别:

    Stage 1: 注视区域裁剪与物体识别
        - 根据 center_x, center_y 裁剪视野区域
        - 识别 exact_gaze_object (注视点正下方物体)
        - 识别 other_objects_in_cropped_area (裁剪区域内其他物体)

    Stage 2: 全帧场景分析
        - 识别 other_objects_outside_fov (视野外物体)
        - 生成 scene_caption (场景描述)
    """
```

**视野半径计算** (基于人眼周边视觉特性):
```python
# 周边凹视觉 (Perifovea) 半径约 13°
HFOV_deg = 90.0      # 相机水平视场角
r_deg = 13.0         # 周边视觉半径 (度)
px_per_deg = frame_width / HFOV_deg
fov_radius = int(r_deg * px_per_deg)  # 约 180-200 像素
```

**输出数据结构**:
```python
{
    'exact_gaze_object': {
        'object_identity': 'knife',
        'description': 'silver kitchen knife with black handle'
    },
    'other_objects_in_cropped_area': [
        {'object_identity': 'cutting board', ...},
        {'object_identity': 'tomato', ...}
    ],
    'other_objects_outside_fov': [
        {'object_identity': 'sink', ...}
    ],
    'scene_caption': 'Person preparing food in a kitchen...'
}
```

### 2.5 Step 3: QA 生成

**文件**: `pipeline/step3_qa_gen.py`, `pipeline/qa_generation/`

基于注视序列和物体标注自动生成三类任务的 QA 对。

**Scanpath 字典构建**:
```python
def create_scanpath_dictionaries(fixation_dataset, video_actions, dataset='egtea'):
    """
    构建时间→物体的映射字典

    返回: {
        (start_time, end_time): [
            [视野内物体列表],
            [视野外物体列表]
        ],
        ...
    }
    """
```

**QA 生成模块**:
```
qa_generation/
├── past.py      # Past 任务生成
│   ├── generate_scene_reconstruction_qa()    # 场景回忆
│   ├── generate_transition_pattern_qa()      # 注视转移预测
│   ├── generate_next_after_group_qa()        # 序列匹配
│   └── generate_never_gazed_qa()             # 未注视物体识别
│
├── present.py   # Present 任务生成
│   └── Present_object_identity_attribute     # 当前注视物体识别
│
└── future.py    # Future/Proactive 任务生成
    ├── generate_future_action_qa()           # 未来动作预测
    └── generate_object_remind_qa()           # 物体提醒任务
```

---

## 3. 眼动数据工程实现

### 3.1 眼动数据格式适配

StreamGaze 支持 4 种不同格式的眼动数据源：

| 数据集 | 格式 | 坐标系 | FPS | 特殊处理 |
|--------|------|--------|-----|----------|
| EGTEA-Gaze+ | BeGaze TXT | 像素坐标 (960×1280) | 24 | 归一化到 [0,1] |
| Ego4D-Gaze | CSV | 归一化坐标 [0,1] | 30 | 置信度阈值过滤 |
| HoloAssist | 3D 世界坐标 | 米 | 24.46 | 需 3D→2D 投影 |
| EgoExoLearn | NPY | 归一化坐标 [0,1] | 30 | validity 标志过滤 |

### 3.2 注视点可视化

生成带眼动叠加的视频用于模型输入：

```python
# 可视化参数
GAZE_DOT_RADIUS = 6           # 注视点半径
GAZE_DOT_COLOR = (0, 0, 255)  # 红色 (BGR)
FOV_CIRCLE_COLOR = (0, 255, 0) # 绿色视野圈
```

**输出目录结构**:
```
dataset/
├── videos/
│   ├── original_video/      # 原始视频
│   └── gaze_viz_video/      # 带眼动可视化的视频
└── qa/
    ├── past_*.json          # Past 任务 QA
    ├── present_*.json       # Present 任务 QA
    └── proactive_*.json     # Proactive 任务 QA
```

### 3.3 时间对齐机制

```python
def build_frame_ticks(start_ticks, fps, n_frames):
    """
    构建帧→时间戳映射
    使用分数运算避免浮点累积误差
    """
    frac = Fraction(fps).limit_denominator()
    deltas = (np.arange(n_frames) * frac.denominator * (10**7)) // frac.numerator
    return start_ticks + deltas

def nearest_index(sorted_ticks, target):
    """
    二分查找最近时间戳
    用于眼动数据与视频帧对齐
    """
```

---

## 4. 大模型评测框架

### 4.1 评测架构

```
src/
├── eval.py                    # 主评测入口
├── benchmark/
│   ├── Benchmark.py           # 基类
│   ├── StreamingBenchGaze_StreamGaze.py      # Present/Future 任务
│   ├── StreamingBenchGaze_Past_StreamGaze.py # Past 任务
│   └── StreamingBenchRemind_StreamGaze.py    # Proactive 任务
├── model/
│   ├── modelclass.py          # 模型基类
│   ├── GPT4o.py               # GPT-4o 封装
│   ├── Gemini.py              # Gemini 封装
│   ├── ClaudeOpus4.py         # Claude Opus 4 封装
│   ├── InternVL.py            # InternVL 封装
│   ├── Qwen25VL.py            # Qwen2.5-VL 封装
│   ├── ViSpeak.py             # ViSpeak (Streaming 模型)
│   └── ...                    # 更多模型
└── utils/
    ├── data_execution.py      # 数据加载与模型调用
    └── video_execution.py     # 视频处理工具
```

### 4.2 模型接口规范

```python
class Model:
    def Run(self, file, inp, start_time, end_time, question_time,
            omni=False, proactive=False, salience_map_path=None):
        """
        参数:
            file: 视频文件路径
            inp: 输入 prompt
            start_time: 视频片段起始时间 (秒)
            end_time: 视频片段结束时间 (秒)
            question_time: 问题时间点 (秒)
            omni: 是否为全能模式
            proactive: 是否为主动任务
            salience_map_path: 显著性图路径

        返回:
            response: 模型输出文本
        """
        return ""

    def name(self):
        return "ModelName"
```

### 4.3 任务评测逻辑

**Present/Future 任务** (60秒窗口):
```python
class StreamingBenchGaze_StreamGaze(Benchmark):
    def eval(self, data, model, output_path):
        for entry in data:
            for question_data in entry["questions"]:
                question_time = parse_timestamp(question_data["time_stamp"])

                # 使用问题时间点前 60 秒的视频窗口
                start_time = max(question_time - 60, 0)
                end_time = question_time

                response = model.Run(
                    file=video_path,
                    inp=prompt,
                    start_time=start_time,
                    end_time=end_time,
                    question_time=question_time
                )
```

**眼动指令注入**:
```python
GAZE_INSTRUCTION = '''In this video, the green dot represents the gaze point
(where the person is looking), and the red circle represents the field of
view (FOV) area.

'''

# 使用带眼动可视化的视频时自动添加
if self.use_gaze_instruction:
    prompt = GAZE_INSTRUCTION + prompt
```

### 4.4 多帧传递机制详解

StreamGaze 针对不同类型的模型采用了不同的多帧传递策略。

#### 4.4.1 API 模型 (GPT-4o) - 固定帧数采样

**文件**: `src/model/GPT4o.py`

GPT-4o 使用 **固定 16 帧均匀采样** 策略：

```python
def extract_frames(self, video_path, start_time, end_time, max_frames=32):
    """使用 decord 从视频片段提取帧"""
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame

    # 均匀采样：如果总帧数 > max_frames，则均匀取样
    if total_frames > max_frames:
        indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)
    else:
        indices = list(range(start_frame, end_frame))

    frames = vr.get_batch(indices).asnumpy()
    return frames

def GPT4o_Run(self, file, inp, start_time, end_time, ...):
    # 始终提取 16 帧，不论视频时长
    max_frames = 16
    frames = self.extract_frames(file, start_time, end_time, max_frames)

    # 每帧转换为 Base64 编码的 JPEG 图片
    content = []
    for frame in frames:
        img_base64 = self.encode_image_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        })

    # 添加文本 prompt
    content.append({"type": "text", "text": inp})

    # 调用 OpenAI API
    response = self.client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=1536
    )
```

**关键实现细节**:
- 使用 `decord` 库进行高效视频解码
- 图片编码: JPEG 格式，质量 85%
- 图片尺寸限制: 最大 2048 像素（自动缩放）
- **60 秒视频 → 16 帧 → 约每 3.75 秒一帧**

#### 4.4.2 开源模型 (Qwen2.5-VL) - 动态 FPS 视频输入

**文件**: `src/model/Qwen25VL.py`

Qwen2.5-VL 支持 **原生视频输入**，采用动态 FPS 策略：

```python
def Qwen25VL_Run(self, file, inp, start_time, end_time, ...):
    duration = end_time - start_time

    # 根据视频时长动态调整 FPS
    if duration <= 30:
        fps = 1.0   # 短视频：每秒 1 帧 → 最多 30 帧
    elif duration <= 60:
        fps = 0.5   # 60 秒视频：每秒 0.5 帧 → 约 30 帧
    elif duration <= 300:
        fps = 0.2   # 5 分钟视频：每秒 0.2 帧 → 约 60 帧
    else:
        fps = 0.1   # 长视频：每秒 0.1 帧

    # 使用 ffmpeg 裁剪视频片段
    temp_video = self.create_video_segment(file, start_time, end_time)

    # 构建 Qwen2.5-VL 消息格式
    content = [
        {
            "type": "video",
            "video": f"file://{temp_video}",
            "fps": fps,  # Qwen2.5-VL 原生支持 fps 参数
        },
        {"type": "text", "text": inp}
    ]

    # 使用 qwen_vl_utils 处理视频
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = self.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )
```

**关键实现细节**:
- 使用 ffmpeg 裁剪指定时间段的视频
- Qwen2.5-VL 原生支持视频文件 + fps 参数
- **60 秒视频，fps=0.5 → 30 帧**
- 处理器自动按 fps 采样帧

#### 4.4.3 帧采样对比表

| 模型类型 | 采样方式 | 60s 视频帧数 | 编码格式 |
|---------|---------|-------------|---------|
| GPT-4o | 固定 16 帧均匀采样 | 16 帧 | Base64 JPEG 多图 |
| Gemini | 类似 GPT-4o | 16 帧 | Base64 多图 |
| Qwen2.5-VL | 动态 FPS (0.5 fps) | ~30 帧 | 原生视频文件 |
| InternVL | 固定帧数采样 | 16-32 帧 | PIL Image 列表 |
| ViSpeak | 流式处理 | 逐帧 | 实时推理 |

#### 4.4.4 视频片段裁剪

**文件**: `src/utils/video_execution.py`

```python
def split_video(video_file, start_time, end_time):
    """使用 ffmpeg 裁剪视频片段"""
    output_file = f"{video_name}_{start_time}_{end_time}.mp4"

    ffmpeg.input(video_file, ss=int(start_time)) \
          .output(output_file, t=(int(end_time) - int(start_time)),
                  vcodec='libx264', acodec='aac') \
          .run()

    return output_file
```

`★ Insight ─────────────────────────────────────`
- **API 模型 (GPT-4o/Gemini/Claude)**: 不支持视频输入，必须提取帧作为多张图片传入
- **开源模型 (Qwen2.5-VL)**: 原生支持视频文件，通过 fps 参数控制采样密度
- **60 秒窗口 + 16 帧** ≈ 每 3.75 秒一帧，这是 API 模型的标准配置
- **不是每秒一帧**：StreamGaze 采用均匀采样而非固定间隔采样
`─────────────────────────────────────────────────`

### 4.4 支持的模型列表

| 模型 | 类型 | 备注 |
|------|------|------|
| GPT-4o | API | OpenAI 闭源 |
| Gemini | API | Google 闭源 |
| Claude Opus 4 | API | Anthropic 闭源 |
| Claude Sonnet 4 | API | Anthropic 闭源 |
| InternVL-2/2.5/3.5 | 开源 | 上海 AI Lab |
| Qwen2-VL / Qwen2.5-VL / Qwen3-VL | 开源 | 阿里 |
| LLaVA-OneVision | 开源 | - |
| VideoLLaMA2 | 开源 | - |
| VILA | 开源 | - |
| ViSpeak | 开源 | Streaming 专用模型 |
| MiniCPM-V / MiniCPM-o | 开源 | 面壁智能 |
| LongVA | 开源 | 长视频模型 |
| VideollmOnline | 开源 | 在线视频理解 |
| FlashVstream | 开源 | Streaming 模型 |
| Kangaroo | 开源 | - |

---

## 5. 时序划分与任务设计详解

### 5.1 Past/Present/Future 时序划分机制

StreamGaze 使用 **`question_time` 作为时间锚点** 来划分过去、现在和未来。

#### 5.1.1 核心概念：问题时间点 (question_time)

```
视频时间线: |----[Past]----[Present]----[Future]----|
                          ↑
                    question_time (锚点)
```

**关键理解**：
- `question_time` 是 QA 生成时标注的时间戳，表示"问题发生的时刻"
- 模型只能看到 `[question_time - 60s, question_time]` 这个窗口内的视频
- 这是 **离线模拟** 流式处理，不是真正的在线推理

#### 5.1.2 评测时的视频窗口

```python
# Present/Future 任务评测逻辑
question_time = parse_timestamp(question_data["time_stamp"])  # 例如 90s

# 模型只能看到问题时间点前 60 秒的视频
start_time = max(question_time - 60, 0)  # 30s
end_time = question_time                  # 90s

# 模型输入: [30s, 90s] 共 60 秒视频
```

#### 5.1.3 Past/Present/Future 的实际含义

| 类别 | 含义 | 模型视角 | 举例 (question_time=90s) |
|------|------|---------|-------------------------|
| **Past** | 需要回忆视频早期内容 | 窗口内的早期部分 [30s-60s] | "你之前看过哪些物体？" |
| **Present** | 当前正在发生的事情 | 窗口末尾附近 [85s-90s] | "你现在看的是什么？" |
| **Future** | 预测即将发生的事情 | 需要推理 question_time 之后 | "接下来你会做什么？" |

`★ Insight ─────────────────────────────────────`
**"未来"如何预测？**
- QA 生成时使用了 **完整视频的知识**（包括 question_time 之后的内容）
- 但评测时只给模型 [t-60, t] 窗口的视频
- 模型需要根据历史模式和当前状态 **推理** 未来
- 这不是"预知"，而是考察模型的 **时序推理能力**
`─────────────────────────────────────────────────`

#### 5.1.4 离线模拟 vs 真正在线

```
┌────────────────────────────────────────────────────────────────┐
│ StreamGaze 是"离线模拟流式"，不是真正的在线处理                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  真正在线模式:                                                  │
│    视频逐帧到达 → 模型实时处理 → 实时输出                        │
│                                                                │
│  StreamGaze 模拟模式:                                          │
│    完整视频 → 裁剪 [t-60, t] → 一次性传入模型 → 输出             │
│                                                                │
│  两者区别:                                                      │
│    - 在线模式：模型有持续的记忆状态                              │
│    - 模拟模式：每次评测都是独立的，无跨问题记忆                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Past Tasks (记忆与时序回忆)

模型需要记忆并推理视频早期发生的事件。

| 任务 | 代码 | 描述 |
|------|------|------|
| Scene Recall (SR) | `generate_scene_reconstruction_qa()` | 回忆用户交互过的物体 |
| Object Transition Prediction (OTP) | `generate_transition_pattern_qa()` | 基于历史注视模式预测下一个物体 |
| Gaze Sequence Matching (GSM) | `generate_next_after_group_qa()` | 匹配注视序列模式 |
| Non-Fixated Objects Identification (NFI) | `generate_never_gazed_qa()` | 识别出现但从未被注视的物体 |

#### 5.2.1 Past 任务示例

**Scene Recall (SR) - 场景回忆**:
```json
{
  "question": "Earlier in the video, what objects did the user interact with while looking at them?",
  "answer": "knife, cutting board, tomato, bowl",
  "time_stamp": "00:01:30",
  "task_type": "scene_recall"
}
```

**Object Transition Prediction (OTP) - 注视转移预测**:
```json
{
  "question": "Based on the user's gaze pattern, after looking at the 'knife', what object did they typically look at next?",
  "options": ["cutting board", "sink", "stove", "refrigerator"],
  "answer": "cutting board",
  "time_stamp": "00:02:00",
  "task_type": "transition_prediction"
}
```

**Gaze Sequence Matching (GSM) - 序列匹配**:
```json
{
  "question": "The user looked at: knife → cutting board → tomato. What did they look at after this sequence?",
  "options": ["bowl", "sink", "pan", "spoon"],
  "answer": "bowl",
  "time_stamp": "00:01:45"
}
```

**Non-Fixated Objects Identification (NFI) - 未注视物体识别**:
```json
{
  "question": "Which of the following objects appeared in the scene but was NEVER gazed at by the user?",
  "options": ["knife", "sink", "cutting board", "tomato"],
  "answer": "sink",
  "time_stamp": "00:02:30"
}
```

### 5.3 Present Tasks (实时感知与推理)

模型需要基于当前眼动信号理解正在发生的事情。

| 任务 | 代码 | 描述 |
|------|------|------|
| Object Identification (Easy/Hard) | `Present_object_identity_attribute` | 识别当前注视物体 |
| Object Attribute Recognition (OAR) | 同上 | 描述注视物体的属性 |
| Future Action Prediction (FAP) | `generate_future_action_qa()` | 预测即将执行的动作 |

#### 5.3.1 Present 任务示例

**Object Identification Easy - 简单物体识别**:
```json
{
  "question": "What object is the user currently looking at?",
  "options": ["knife", "spoon", "fork", "chopsticks"],
  "answer": "knife",
  "time_stamp": "00:01:15",
  "difficulty": "easy"
}
```
> Easy 难度：选项是同类物体（都是餐具），但视觉差异明显

**Object Identification Hard - 困难物体识别**:
```json
{
  "question": "What specific type of container is the user looking at?",
  "options": ["glass bowl", "plastic container", "ceramic dish", "metal pot"],
  "answer": "glass bowl",
  "time_stamp": "00:01:45",
  "difficulty": "hard"
}
```
> Hard 难度：需要识别物体的具体属性（材质、类型）

**Object Attribute Recognition (OAR) - 属性识别**:
```json
{
  "question": "Describe the color and material of the object the user is currently gazing at.",
  "answer": "silver metallic knife with a black plastic handle",
  "time_stamp": "00:01:20"
}
```

**Future Action Prediction (FAP) - 未来动作预测**:
```json
{
  "question": "Based on the user's current gaze on the tomato and the knife nearby, what action are they most likely to perform next?",
  "options": ["cut the tomato", "wash the tomato", "eat the tomato", "put away the tomato"],
  "answer": "cut the tomato",
  "time_stamp": "00:01:30"
}
```

### 5.4 Proactive Tasks (预期与主动提醒)

最具挑战性的任务类别，模型需要预测未来并主动响应。

| 任务 | 代码 | 描述 |
|------|------|------|
| Gaze-Triggered Alert (GTA) | `generate_object_remind_qa()` | 当用户注视特定目标时发出提醒 |
| Object Appearance Alert (OAA) | 同上 | 当目标物体出现在场景中时发出提醒 |

#### 5.4.1 Proactive 任务示例

**Gaze-Triggered Alert Easy (GTA-E)**:
```json
{
  "instruction": "Alert the user when they look at the 'stove'",
  "target_object": "stove",
  "trigger_condition": "gaze",
  "expected_alert_time": "00:02:15",
  "alert_message": "Reminder: You are looking at the stove. Don't forget to turn it off after cooking.",
  "difficulty": "easy"
}
```
> Easy：目标物体在用户视野内 (FOV)，容易检测到注视

**Gaze-Triggered Alert Hard (GTA-H)**:
```json
{
  "instruction": "Alert the user when they look at the 'fire extinguisher'",
  "target_object": "fire extinguisher",
  "trigger_condition": "gaze",
  "expected_alert_time": "00:03:45",
  "difficulty": "hard"
}
```
> Hard：目标物体在视野外 (outside FOV)，需要模型持续追踪并识别

**Object Appearance Alert (OAA)**:
```json
{
  "instruction": "Alert the user when a 'cat' appears in the scene",
  "target_object": "cat",
  "trigger_condition": "appearance",
  "expected_alert_time": "00:04:20",
  "alert_message": "Notice: A cat has entered the kitchen area."
}
```

#### 5.4.2 Proactive 任务的生成逻辑

```python
def generate_object_remind_qa(scanpath_dict, ...):
    """
    生成 Proactive 任务

    Easy 模式 (target_in_fov=True):
        - 从 FOV 内物体中选择目标
        - 模型容易在注视时检测到

    Hard 模式 (target_in_fov=False):
        - 从 FOV 外物体中选择目标
        - 模型需要持续监控整个场景

    时间约束:
        - question_time 与 target 出现时间间隔 2-120 秒
        - 避免目标出现太早（太简单）或太晚（超出窗口）
    """
```

---

## 6. 运行指南

### 6.1 数据准备

```bash
# 从 HuggingFace 下载数据集
# 放置到以下结构:
StreamGaze/
├── dataset/
│   ├── videos/
│   │   ├── original_video/
│   │   └── gaze_viz_video/
│   └── qa/
│       ├── past_*.json
│       ├── present_*.json
│       └── proactive_*.json
```

### 6.2 评测命令

```bash
# 评测 Qwen2.5-VL (带眼动可视化)
bash scripts/qwen25vl.sh --use_gaze_instruction

# 评测 GPT-4o
bash scripts/gpt4o.sh --use_gaze_instruction

# 评测 ViSpeak (无眼动可视化)
bash scripts/vispeak.sh
```

### 6.3 添加新模型

1. 实现模型封装:
```python
# src/model/YourModel.py
from model.modelclass import Model

class YourModel(Model):
    def __init__(self):
        # 加载模型
        self.model = ...

    def Run(self, file, inp, start_time, end_time, question_time, **kwargs):
        # 处理视频和生成回答
        return "Your model's response"

    def name(self):
        return "YourModel"
```

2. 注册模型:
```python
# src/eval.py
elif args.model_name == "YourModel":
    from model.YourModel import YourModel
    model = YourModel()
```

---

## 7. 技术亮点与启示

### 7.1 眼动数据的创新应用

`★ Insight ─────────────────────────────────────`
- **显式注意力信号**: 将人类眼动作为显式输入，而非仅依赖视觉显著性
- **周边视觉建模**: 使用 13° 视角的 FOV 裁剪模拟人眼周边凹视觉
- **时序注视序列**: Scanpath 不仅记录"看了什么"，还保留"什么时候看的"
`─────────────────────────────────────────────────`

### 7.2 流式视频理解的设计

`★ Insight ─────────────────────────────────────`
- **60秒滑动窗口**: Present/Future 任务使用 [t-60, t] 窗口，模拟实时处理
- **渐进式上下文**: Past 任务需要访问更长的历史，测试模型记忆能力
- **主动式交互**: Proactive 任务要求模型在正确时机主动发出提醒
`─────────────────────────────────────────────────`

### 7.3 多阶段 VLM Pipeline

`★ Insight ─────────────────────────────────────`
- **Stage 1 (InternVL-38B)**: 物体识别与场景标注
- **Stage 2 (Qwen3VL-30B)**: QA 质量验证与过滤
- **两阶段设计**: 分离"数据生成"与"质量控制"，提高标注准确性
`─────────────────────────────────────────────────`

---

## 8. 对 Long Video Understanding 研究的启示

1. **眼动作为先验**: 眼动数据可以作为视频理解的强先验，指导模型关注重要区域
2. **时序推理评测**: StreamGaze 提供了完整的时序推理评测框架，可借鉴其 Past/Present/Future 任务设计
3. **主动式 AI**: Proactive 任务的设计对于 AR/VR 助手等应用场景有重要参考价值
4. **数据生成 Pipeline**: 其自动化 QA 生成流程可复用于其他视频理解 benchmark 的构建

---

## 参考资料

- **论文**: https://arxiv.org/abs/2512.01707
- **代码**: https://github.com/daeunni/StreamGaze
- **数据集**: HuggingFace (链接待公开)
- **相关工作**: StreamingBench, EGTEA-Gaze+, Ego4D, HoloAssist, EgoExoLearn

---

## 9. 数据集构成详解

### 9.1 StreamGaze Benchmark 总体统计

| 指标 | 数值 |
|------|------|
| 视频总数 | **285** |
| QA 对总数 | **8,521** |
| 任务类别 | **10 种** (4 Past + 4 Present + 2 Proactive) |
| 支持数据集 | EGTEA-Gaze+, Ego4D-Gaze, HoloAssist, EgoExoLearn |

### 9.2 源数据集详细信息

StreamGaze 从 4 个公开的第一人称视频数据集中提取并处理数据：

#### 9.2.1 EGTEA-Gaze+ (Georgia Tech)

| 属性 | 详情 |
|------|------|
| **总时长** | 28 小时 |
| **视频数量** | 86 个 session，32 个被试 |
| **分辨率** | 1280×960 (HD) |
| **帧率** | 24 FPS (视频) / 30 Hz (眼动) |
| **场景** | 厨房烹饪 |
| **活动类型** | 7 种烹饪任务 |

**具体活动列表**:
1. American Breakfast (美式早餐)
2. Pizza (披萨)
3. Snack (零食)
4. Greek Salad (希腊沙拉)
5. Pasta Salad (意面沙拉)
6. Turkey Sandwich (火鸡三明治)
7. Cheese Burger (芝士汉堡)

**数据特点**:
- 最成熟的第一人称烹饪数据集
- 包含帧级动作标注 (71 个动作类别)
- 提供手部分割掩码 (14K 帧)
- BeGaze 格式的眼动数据

#### 9.2.2 Ego4D-Gaze (Meta AI)

| 属性 | 详情 |
|------|------|
| **总时长** | 3,670 小时 (Ego4D 全集) |
| **参与者** | 931 人，来自 74 个地点 |
| **场景** | 日常生活多场景 |
| **眼动格式** | CSV (归一化坐标 + 置信度) |

**场景覆盖**:
- 家庭场景 (Household)
- 户外场景 (Outdoor)
- 工作场所 (Workplace)
- 休闲场景 (Leisure)
- 社交互动 (Social)

**数据特点**:
- 全球最大规模的第一人称视频数据集
- 多样化的日常活动
- 跨文化、跨地域采集

#### 9.2.3 HoloAssist (Microsoft Research)

| 属性 | 详情 |
|------|------|
| **设备** | Microsoft HoloLens 2 |
| **场景** | 物理操作任务 + AR 辅助 |
| **特殊性** | 3D 眼动追踪 (需投影到 2D) |
| **交互模式** | 双人协作 (执行者 + 指导者) |

**任务类型**:
- 物理组装任务
- 设备维修操作
- 实验室操作流程

**数据特点**:
- 首个 AR 辅助场景下的第一人称数据集
- 包含错误检测、干预预测等标注
- 3D 手部姿态数据
- 需要 `step0_gaze_projection.py` 将 3D 眼动投影到 2D

#### 9.2.4 EgoExoLearn (OpenGVLab)

| 属性 | 详情 |
|------|------|
| **第一人称视频** | 432 个视频，共 96.5 小时 |
| **示范视频** | 315 个视频，共 23.5 小时 |
| **场景** | 日常任务 + 实验室任务 |
| **眼动格式** | NPY (归一化坐标 + validity) |

**任务分类**:

| 类别 | 任务类型 |
|------|---------|
| **日常任务** (5种) | 烹饪、手工制作、家务等 |
| **实验室任务** (3种) | 固相肽合成、化学实验、生物实验等 |

**数据特点**:
- Ego-Exo 视角对齐（第一人称 + 第三人称）
- 高质量眼动数据
- 程序性任务学习场景

### 9.3 数据集对比总览

| 数据集 | 视频时长 | 主要场景 | 眼动格式 | 特殊标注 |
|--------|---------|---------|---------|---------|
| EGTEA-Gaze+ | 28h | 厨房烹饪 | BeGaze TXT | 动作标签、手部掩码 |
| Ego4D-Gaze | 3,670h | 日常生活 | CSV | 多任务 benchmark |
| HoloAssist | - | AR 辅助任务 | 3D 世界坐标 | 错误检测、干预预测 |
| EgoExoLearn | 96.5h | 日常+实验室 | NPY | Ego-Exo 对齐 |

### 9.4 StreamGaze 中各数据集的贡献

```
┌─────────────────────────────────────────────────────────────────┐
│                  StreamGaze 数据集构成                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  EGTEA-Gaze+                                                   │
│  ├── 场景: 厨房烹饪 (7种食谱)                                    │
│  ├── 典型视频长度: 3-10 分钟                                     │
│  └── 贡献任务: Past/Present/Future 全类型                       │
│                                                                 │
│  Ego4D-Gaze                                                    │
│  ├── 场景: 日常生活多场景                                        │
│  ├── 典型视频长度: 5-30 分钟                                     │
│  └── 贡献任务: 多样化场景理解                                    │
│                                                                 │
│  HoloAssist                                                    │
│  ├── 场景: AR 辅助物理任务                                       │
│  ├── 典型视频长度: 5-20 分钟                                     │
│  └── 贡献任务: Proactive 提醒任务                               │
│                                                                 │
│  EgoExoLearn                                                   │
│  ├── 场景: 程序性任务学习                                        │
│  ├── 典型视频长度: 5-15 分钟                                     │
│  └── 贡献任务: 技能学习相关任务                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.5 任务分布统计

| 任务类别 | 任务名称 | 描述 |
|---------|---------|------|
| **Past** (4) | Scene Recall | 回忆交互过的物体 |
|  | Object Transition Prediction | 预测注视转移模式 |
|  | Gaze Sequence Matching | 匹配注视序列 |
|  | Non-Fixated Objects ID | 识别未注视物体 |
| **Present** (4) | Object ID (Easy) | 识别注视物体 (简单) |
|  | Object ID (Hard) | 识别注视物体 (困难) |
|  | Object Attribute Recognition | 描述物体属性 |
|  | Future Action Prediction | 预测下一个动作 |
| **Proactive** (2) | Gaze-Triggered Alert | 注视触发提醒 |
|  | Object Appearance Alert | 物体出现提醒 |

`★ Insight ─────────────────────────────────────`
**数据集设计亮点**:
- **多源融合**: 4 个数据集覆盖厨房、日常、AR、实验室多种场景
- **真实眼动**: 所有数据都包含真实人类眼动追踪数据
- **任务多样性**: 从简单物体识别到复杂主动提醒，难度梯度清晰
- **程序性任务**: 重点关注有步骤的操作流程，适合测试时序推理
`─────────────────────────────────────────────────`
