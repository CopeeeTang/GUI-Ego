激活虚拟环境
cd /home/v-tangxin/GUI
source ml_env/bin/activate
python3 instead of python

Always response in Chinese

GPU: NVIDIA A100 80GB PCIe
CUDA驱动版本: 13.0 (Driver 580.95.05)
PyTorch CUDA版本: 12.8 (完全兼容)

常用命令
数据生成
python3 -m agent.src.pipeline --compare --scenes general --limit 1 --verbose

前端展示
./ml_env/bin/python3 -m agent.preview.server  --port 8000

会话历史保存
当对话内容丰富（多轮探索、重要决策、实验结果）且 context 使用较多时，主动提醒用户运行 /save-session 保存行动路线摘要。
如果用户即将结束会话或切换话题，也应建议保存。
保存目录: docs/history/{project_name}/round_{N}.md

