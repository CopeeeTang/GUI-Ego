"""ProAssist 环境验证脚本
验证模型加载、数据加载和推理是否正常工作。
"""
import os
os.environ["DATA_ROOT_DIR"] = "/home/v-tangxin/GUI/data/ProAssist"

import sys
sys.path.insert(0, "/home/v-tangxin/GUI/temp/ProAssist")

import torch
from mmassist.model import build_from_checkpoint
from mmassist.model.configuration_proact import ProActLlamaConfig
from mmassist.data import build_eval_datasets
from mmassist.configs import ModelArguments

MODEL_PATH = "/home/v-tangxin/GUI/data/ProAssist/ProAssist-Model-L4096-I1"
DATA_ROOT = os.path.join(os.environ["DATA_ROOT_DIR"], "processed_data")

print("=" * 60)
print("ProAssist 环境验证")
print("=" * 60)

# Step 1: Load model with sdpa attention
print("\n[1/3] 加载模型...")
model_args = ModelArguments(attn_implementation="sdpa")
model, tokenizer = build_from_checkpoint(MODEL_PATH, model_args=model_args)
print(f"  模型类型: {type(model).__name__}")
print(f"  设备: {model.device}")
print(f"  数据类型: {model.dtype}")
print(f"  注意力实现: {model.config.attn_implementation}")
print(f"  最大序列长度: {model.config.max_seq_len}")

# Step 2: Load dataset
print("\n[2/3] 加载 WTAG 数据集...")
model_config = model.config.to_dict()
datasets = build_eval_datasets(
    eval_datasets="wtag/dialog-klg-sum_val_L4096_I1",
    data_root_dir=DATA_ROOT,
    **model_config
)
dataset_name = list(datasets.keys())[0]
dataset = datasets[dataset_name]
print(f"  数据集: {dataset_name}")
print(f"  样本数: {len(dataset)}")

# Step 3: Run inference on first sample
print("\n[3/3] 运行推理（前10秒）...")
from mmassist.eval.runners import StreamInferenceRunner

runner = StreamInferenceRunner.build(
    eval_name="verify",
    model=model,
    tokenizer=tokenizer,
    fps=2,
    not_talk_threshold=0.5,
    eval_max_seq_len=4096,
)

video = dataset[0]
print(f"  视频: {video['video_uid']}, 帧范围: [{video['start_frame_idx']}, {video['end_frame_idx']})")

result = runner.run_inference_on_video(video, verbose=True, max_time=10)
n_predictions = len(result['predictions'])
n_with_text = sum(1 for p in result['predictions'] if p.gen.strip())

print(f"\n{'=' * 60}")
print(f"验证成功!")
print(f"  总帧输出: {n_predictions}")
print(f"  有文本输出的帧: {n_with_text}")
print(f"  元数据: {result['metadata']}")
print(f"{'=' * 60}")
