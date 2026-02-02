#!/bin/bash
# Multi-LLM + Multi-Strategy 批量测试脚本
# 5 个模型 × 3 个策略 = 15 组测试

cd /home/v-tangxin/GUI
source ml_env/bin/activate

LIMIT=10

echo "=========================================="
echo "Multi-LLM Pipeline Batch Test"
echo "Models: 5, Strategies: 3, Samples: $LIMIT"
echo "=========================================="

# ==========================================
# GPT-4o (Azure)
# ==========================================
echo ""
echo ">>> [1/15] GPT-4o + v1_baseline"
python3 -m agent.src.pipeline --model azure:gpt-4o --strategy v1_baseline --limit $LIMIT

echo ""
echo ">>> [2/15] GPT-4o + v2_google_gui"
python3 -m agent.src.pipeline --model azure:gpt-4o --strategy v2_google_gui --limit $LIMIT

echo ""
echo ">>> [3/15] GPT-4o + v3_with_visual"
python3 -m agent.src.pipeline --model azure:gpt-4o --strategy v3_with_visual --enable-visual --limit $LIMIT

# ==========================================
# Gemini-3-Flash
# ==========================================
echo ""
echo ">>> [4/15] Gemini-3-Flash + v1_baseline"
python3 -m agent.src.pipeline --model gemini:gemini-3-flash --strategy v1_baseline --limit $LIMIT

echo ""
echo ">>> [5/15] Gemini-3-Flash + v2_google_gui"
python3 -m agent.src.pipeline --model gemini:gemini-3-flash --strategy v2_google_gui --limit $LIMIT

echo ""
echo ">>> [6/15] Gemini-3-Flash + v3_with_visual"
python3 -m agent.src.pipeline --model gemini:gemini-3-flash --strategy v3_with_visual --enable-visual --limit $LIMIT

# ==========================================
# Gemini-3-Pro-High
# ==========================================
echo ""
echo ">>> [7/15] Gemini-3-Pro-High + v1_baseline"
python3 -m agent.src.pipeline --model gemini:gemini-3-pro-high --strategy v1_baseline --limit $LIMIT

echo ""
echo ">>> [8/15] Gemini-3-Pro-High + v2_google_gui"
python3 -m agent.src.pipeline --model gemini:gemini-3-pro-high --strategy v2_google_gui --limit $LIMIT

echo ""
echo ">>> [9/15] Gemini-3-Pro-High + v3_with_visual"
python3 -m agent.src.pipeline --model gemini:gemini-3-pro-high --strategy v3_with_visual --enable-visual --limit $LIMIT

# ==========================================
# Claude Sonnet 4.5 (Thinking)
# ==========================================
echo ""
echo ">>> [10/15] Claude-Sonnet-4.5-Thinking + v1_baseline"
python3 -m agent.src.pipeline --model claude:claude-sonnet-4-5-thinking --strategy v1_baseline --limit $LIMIT

echo ""
echo ">>> [11/15] Claude-Sonnet-4.5-Thinking + v2_google_gui"
python3 -m agent.src.pipeline --model claude:claude-sonnet-4-5-thinking --strategy v2_google_gui --limit $LIMIT

echo ""
echo ">>> [12/15] Claude-Sonnet-4.5-Thinking + v3_with_visual"
python3 -m agent.src.pipeline --model claude:claude-sonnet-4-5-thinking --strategy v3_with_visual --enable-visual --limit $LIMIT

# ==========================================
# Claude Opus 4.5
# ==========================================
echo ""
echo ">>> [13/15] Claude-Opus-4.5 + v1_baseline"
python3 -m agent.src.pipeline --model claude:claude-opus-4-5 --strategy v1_baseline --limit $LIMIT

echo ""
echo ">>> [14/15] Claude-Opus-4.5 + v2_google_gui"
python3 -m agent.src.pipeline --model claude:claude-opus-4-5 --strategy v2_google_gui --limit $LIMIT

echo ""
echo ">>> [15/15] Claude-Opus-4.5 + v3_with_visual"
python3 -m agent.src.pipeline --model claude:claude-opus-4-5 --strategy v3_with_visual --enable-visual --limit $LIMIT

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Output directory: /home/v-tangxin/GUI/agent/output/"
echo "=========================================="
