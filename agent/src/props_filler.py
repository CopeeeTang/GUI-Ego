"""
Props 填充器

使用 LLM 根据推荐内容填充组件的 props
"""

import json
import uuid
from typing import Any

from .llm_client import LLMClient
from .data_loader import Recommendation, SceneConfig


# 组件 Props Schema
COMPONENT_SCHEMAS = {
    "map_card": {
        "required": ["title"],
        "properties": {
            "title": "string - 卡片标题",
            "subtitle": "string - 副标题（可选）",
            "markers": "array - 地图标记点列表，每个包含 {type, label, count, distance}",
            "action": "object - 操作按钮 {label, type: 'navigate'|'select'|'dismiss'}",
        },
        "example": {
            "title": "附近共享单车",
            "subtitle": "3辆可用",
            "markers": [{"type": "bike", "label": "哈罗单车", "count": 3, "distance": "50m"}],
            "action": {"label": "导航到最近单车", "type": "navigate"},
        },
    },
    "ar_label": {
        "required": ["text"],
        "properties": {
            "text": "string - 主要文字",
            "subtext": "string - 次要文字（可选）",
            "icon": "string - 图标名称（可选）",
            "anchor": "object - 锚定信息 {type: 'object'|'location', target: string}",
        },
        "example": {
            "text": "清华大学档案馆",
            "subtext": "C楼 · 50m",
            "anchor": {"type": "object", "target": "building"},
        },
    },
    "comparison_card": {
        "required": ["title", "items"],
        "properties": {
            "title": "string - 对比标题",
            "items": "array - 对比项列表，每个包含 {label, value?, score?, highlight?}",
            "recommendation": "string - 推荐结论",
        },
        "example": {
            "title": "橙子新鲜度对比",
            "items": [
                {"label": "盒A", "score": 8.5, "highlight": True},
                {"label": "盒B", "score": 7.2},
            ],
            "recommendation": "盒A 更新鲜，颜色更鲜亮",
        },
    },
    "direction_arrow": {
        "required": ["direction"],
        "properties": {
            "direction": "string - 方向: left/right/forward/back",
            "distance": "string - 距离",
            "destination": "string - 目的地名称",
            "eta": "string - 预计到达时间",
        },
        "example": {
            "direction": "forward",
            "distance": "200m",
            "destination": "学生事务中心",
            "eta": "3分钟",
        },
    },
    "nutrition_card": {
        "required": ["product_name"],
        "properties": {
            "product_name": "string - 产品名称",
            "calories": "number - 热量(kcal)",
            "protein": "string - 蛋白质",
            "carbs": "string - 碳水化合物",
            "fat": "string - 脂肪",
            "sugar": "string - 糖分",
            "health_rating": "integer - 健康评级 1-5",
        },
        "example": {
            "product_name": "炭烧酸奶",
            "calories": 180,
            "protein": "6g",
            "carbs": "25g",
            "sugar": "18g",
            "health_rating": 3,
        },
    },
    "price_calculator": {
        "required": ["items"],
        "properties": {
            "title": "string - 标题",
            "items": "array - 商品列表 {name, price, unit, quantity, unit_price}",
            "total": "number - 总价",
            "recommendation": "string - 推荐建议",
        },
        "example": {
            "title": "性价比对比",
            "items": [
                {"name": "切好的水果", "price": 15.9, "unit": "盒", "unit_price": 15.9},
                {"name": "整颗橙子", "price": 12.8, "unit": "斤", "unit_price": 6.4},
            ],
            "recommendation": "整颗橙子性价比更高",
        },
    },
    "task_card": {
        "required": ["title"],
        "properties": {
            "title": "string - 任务标题",
            "description": "string - 任务描述",
            "due_time": "string - 截止时间",
            "action": "object - 操作 {confirm_label, dismiss_label}",
        },
        "example": {
            "title": "午饭提醒",
            "description": "30分钟后去青青快餐吃午饭",
            "due_time": "12:30",
            "action": {"confirm_label": "设置提醒", "dismiss_label": "取消"},
        },
    },
    "step_card": {
        "required": ["title", "steps"],
        "properties": {
            "title": "string - 流程标题",
            "current_step": "integer - 当前步骤",
            "steps": "array - 步骤列表 {number, instruction, completed}",
        },
        "example": {
            "title": "申请学生档案流程",
            "current_step": 1,
            "steps": [
                {"number": 1, "instruction": "在前台登记", "completed": False},
                {"number": 2, "instruction": "填写申请表", "completed": False},
                {"number": 3, "instruction": "等待叫号", "completed": False},
            ],
        },
    },
    "info_card": {
        "required": ["title", "content"],
        "properties": {
            "title": "string - 卡片标题",
            "subtitle": "string - 副标题（可选）",
            "content": "string - 详细描述内容",
            "image_url": "string - 相关的图片链接（可选）",
            "action": "object - 操作按钮 {label, type: 'open'|'dismiss'}",
        },
        "example": {
            "title": "红酒百科",
            "subtitle": "赤霞珠 (Cabernet Sauvignon)",
            "content": "赤霞珠是世界上最著名的红葡萄品种之一。它以色泽深沉、单宁厚重、香气浓郁（如黑加仑、青椒、雪松）而闻名。通常需要较长时间的陈年以达到最佳风味。",
            "action": {"label": "查看更多", "type": "open"},
        },
    },
}

PROPS_FILLING_PROMPT = """你是一个智能眼镜 UI 内容生成专家。根据 AI 推荐内容，填充 UI 组件的 props。

## 组件类型: {component_type}

## 组件 Schema:
必需字段: {required_fields}
可用属性:
{properties}

## 示例:
```json
{example}
```

## AI 推荐
- 类型: {recommendation_type}
- 内容: {recommendation_content}

## 任务
根据推荐内容，生成该组件的 props。确保：
1. 包含所有必需字段
2. 内容与推荐语义一致
3. 适合智能眼镜的简洁显示

## 输出
返回 JSON 格式的 props 对象（不需要 type 和 id，只需要 props 内容）:
"""


class PropsFiller:
    """Props 填充器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def fill_props(
        self,
        component_type: str,
        recommendation: Recommendation,
    ) -> dict[str, Any]:
        """填充组件 props"""
        schema = COMPONENT_SCHEMAS.get(component_type)
        if not schema:
            raise ValueError(f"No schema defined for component: {component_type}")

        # 构建属性描述
        properties = "\n".join(
            f"- {k}: {v}" for k, v in schema["properties"].items()
        )

        prompt = PROPS_FILLING_PROMPT.format(
            component_type=component_type,
            required_fields=", ".join(schema["required"]),
            properties=properties,
            example=json.dumps(schema["example"], ensure_ascii=False, indent=2),
            recommendation_type=recommendation.type,
            recommendation_content=recommendation.content,
        )

        props = self.llm.complete_json(
            system_prompt="你是一个 UI 内容生成专家，只返回 JSON 格式的 props 对象。",
            user_prompt=prompt,
            temperature=0.5,
        )

        return props

    def generate_component(
        self,
        component_type: str,
        recommendation: Recommendation,
    ) -> dict[str, Any]:
        """生成完整的 A2UI 组件"""
        props = self.fill_props(component_type, recommendation)

        return {
            "type": component_type,
            "id": f"{component_type}_{uuid.uuid4().hex[:8]}",
            "props": props,
            "metadata": {
                "source_recommendation": {
                    "type": recommendation.type,
                    "content": recommendation.content,
                    "time_range": [recommendation.start_time, recommendation.end_time],
                },
                "generated": True,
            },
        }
