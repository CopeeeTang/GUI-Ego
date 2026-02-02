激活虚拟环境
cd /home/v-tangxin/GUI
source ml_env/bin/activate
python3 instead of python

GPU: NVIDIA A100 80GB PCIe
CUDA驱动版本: 13.0 (Driver 580.95.05)
PyTorch CUDA版本: 12.8 (完全兼容)

常用命令
数据生成
python3 -m agent.src.pipeline --compare --scenes general --limit 1 --verbose

前端展示
./ml_env/bin/python3 -m agent.preview.server --port 8000
--reload实现自动重载
