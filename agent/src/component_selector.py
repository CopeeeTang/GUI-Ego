"""
组件选择器

使用 LLM 根据推荐内容选择最合适的 UI 组件
"""

import json
from typing import Any

from .llm_client import LLMClient
from .data_loader import Recommendation, SceneConfig


# 组件定义
COMPONENT_DEFINITIONS = {
    "navigation": {
        "map_card": {
            "description": "地图卡片，用于显示位置、导航路线、附近的兴趣点（如共享单车、建筑）",
            "use_cases": ["找共享单车", "导航到目的地", "显示附近设施", "估算距离/时间"],
        },
        "ar_label": {
            "description": "AR 标签，悬浮在物理对象上方显示信息",
            "use_cases": ["识别建筑名称", "标注地标", "显示方向指示", "物体识别结果"],
        },
        "direction_arrow": {
            "description": "方向指引箭头，显示前进方向和距离",
            "use_cases": ["步行导航", "指示转弯方向", "显示目的地距离"],
        },
        "comparison_card": {
            "description": "对比卡片，并排比较两个或多个选项",
            "use_cases": ["比较步行vs骑车时间", "比较路线选项", "决策支持"],
        },
    },
    "shopping": {
        "comparison_card": {
            "description": "对比卡片，并排比较商品属性",
            "use_cases": ["比较商品新鲜度", "比较价格", "比较营养成分", "决策支持"],
        },
        "nutrition_card": {
            "description": "营养信息卡片，显示食品的营养数据",
            "use_cases": ["显示热量", "显示蛋白质/碳水/脂肪", "显示成分列表", "健康评级"],
        },
        "price_calculator": {
            "description": "价格计算器，计算单价、总价、性价比",
            "use_cases": ["计算单价", "比较性价比", "估算总价"],
        },
        "ar_label": {
            "description": "AR 标签，叠加在商品上显示信息",
            "use_cases": ["显示商品名称", "显示价格", "标注推荐商品"],
        },
    },
}

COMPONENT_SELECTION_PROMPT = """你是一个智能眼镜 UI 生成专家。根据用户场景和 AI 推荐内容，选择最合适的 UI 组件。

## 当前场景: {scene_name}

## 可用组件:
{component_list}

## 任务
分析以下 AI 推荐内容，选择最合适的 UI 组件类型。

## AI 推荐
- 类型: {recommendation_type}
- 内容: {recommendation_content}

## 输出要求
返回 JSON 格式:
{{
    "selected_component": "组件类型名称",
    "reasoning": "选择理由（简短）",
    "confidence": 0.0-1.0 的置信度
}}
"""


class ComponentSelector:
    """组件选择器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def select_component(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
    ) -> dict[str, Any]:
        """选择最合适的组件"""
        # 获取场景组件定义
        scene_components = COMPONENT_DEFINITIONS.get(scene.name, {})
        if not scene_components:
            raise ValueError(f"No components defined for scene: {scene.name}")

        # 构建组件列表描述
        component_list = []
        for comp_name, comp_info in scene_components.items():
            component_list.append(
                f"- **{comp_name}**: {comp_info['description']}\n"
                f"  适用场景: {', '.join(comp_info['use_cases'])}"
            )

        prompt = COMPONENT_SELECTION_PROMPT.format(
            scene_name=scene.name,
            component_list="\n".join(component_list),
            recommendation_type=recommendation.type,
            recommendation_content=recommendation.content,
        )

        result = self.llm.complete_json(
            system_prompt="你是一个 UI 组件选择专家，只返回 JSON 格式的结果。",
            user_prompt=prompt,
            temperature=0.3,
        )

        return result
