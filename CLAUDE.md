## 服务器配置
激活虚拟环境
cd /home/v-tangxin/GUI
source ml_env/bin/activate
python3 instead of python

## 规范
Always response in Chinese

GPU: NVIDIA A100 80GB PCIe
CUDA驱动版本: 13.0 (Driver 580.95.05)
PyTorch CUDA版本: 12.8 (完全兼容)

## API & 基础设施
- GPT-4o 通过 Azure 本地代理: `http://52.151.57.21:9999`
- 永远不要直接调用 OpenAI/Google/Anthropic 的官方 API endpoint
- 发起 API 请求前先检查现有代码中的 proxy/endpoint 配置

## Python 规范
- CLI 参数用连字符: `--edge-model`, `--learning-rate`
- Python 变量用下划线: `edge_model`, `learning_rate`
- 修改 argparse 定义时，必须检查所有引用该参数的位置（包括 `run()` 函数默认值）
- 编辑包含花括号的 Python 字符串（f-string, template string）时，确认转义正确
- 编辑 Python 文件后，心理检查一遍语法正确性（花括号、引号匹配、缩进）

## 实验执行
- 运行实验前必须验证：
  1. `python3 script.py --help` 确认所有 CLI 参数存在且格式正确
  2. 所需的环境变量和模型路径已设置
  3. 先做一次 dry-run 或小规模测试确保无 import/参数错误
- 二进制文件（.docx, .xlsx, .pptx）用对应 Python 库读取，不要当文本读


## 常用命令
数据生成
python3 -m agent.src.pipeline --compare --scenes general --limit 1 --verbose

前端展示
./ml_env/bin/python3 -m agent.preview.server  --port 8000

## Code Review（Codex）
- 使用 `/codex` skill 或直接调用 `codex exec` 进行代码审查和 bug 检查
- 推荐工作流：Claude 分析定位问题 → Codex 执行修改 → Claude 验证结果
- 常用命令：
  - 只读审查: `codex exec -s read-only "review agent/src/ for bugs and code quality"`
  - 自动修复: `codex exec --full-auto "fix bugs in agent/src/ and run py_compile after each fix"`
  - JSON 报告: `codex exec -s read-only --json -o review.json "code review report"`
- 审查重点：参数解析正确性、template string 转义、错误处理、API 调用模式

## 会话历史保存
当对话内容丰富（多轮探索、重要决策、实验结果）且 context 使用较多时，主动提醒用户运行 /save-session 保存行动路线摘要。
如果用户即将结束会话或切换话题，也应建议保存。
保存目录: docs/history/{project_name}/round_{N}.md
