"""
组件选择器

使用 LLM 根据推荐内容选择最合适的 UI 组件
"""

import json
from typing import Any

from .llm_client import LLMClient
from .data_loader import Recommendation, SceneConfig


# A2UI 原子组件定义 - 与 Preview 渲染器兼容
COMPONENT_DEFINITIONS = {
    "navigation": {
        "Card": {
            "description": "容器卡片，用于组合显示导航信息、地点详情、路线选项",
            "use_cases": ["显示目的地信息", "展示路线选项", "呈现附近地点", "导航卡片容器"],
        },
        "Text": {
            "description": "文本元素，显示标题、描述、距离、时间等文字信息",
            "use_cases": ["显示地点名称", "显示距离时间", "路线描述", "状态提示"],
        },
        "Button": {
            "description": "交互按钮，触发导航、选择路线、查看详情等操作",
            "use_cases": ["开始导航", "选择路线", "查看详情", "取消操作"],
        },
        "Icon": {
            "description": "图标元素，表示方向、地点类型、交通方式等",
            "use_cases": ["方向箭头", "地点图标", "交通工具图标", "状态指示"],
        },
        "Badge": {
            "description": "标签徽章，显示距离、时间、状态等简短信息",
            "use_cases": ["显示距离", "显示预计时间", "状态标签", "类型标记"],
        },
        "Row": {
            "description": "水平布局容器，横向排列子元素",
            "use_cases": ["并排显示选项", "图标+文字组合", "操作按钮组"],
        },
        "Column": {
            "description": "垂直布局容器，纵向排列子元素",
            "use_cases": ["信息列表", "多行内容", "分步指引"],
        },
    },
    "shopping": {
        "Card": {
            "description": "容器卡片，用于显示商品信息、价格比较、营养数据",
            "use_cases": ["商品信息卡", "价格比较卡", "营养信息卡", "推荐商品卡"],
        },
        "Text": {
            "description": "文本元素，显示商品名称、价格、描述、营养成分等",
            "use_cases": ["商品名称", "价格显示", "营养数值", "描述文字"],
        },
        "Button": {
            "description": "交互按钮，添加购物车、查看详情、比较商品等",
            "use_cases": ["添加购物车", "查看详情", "比较选中", "立即购买"],
        },
        "Badge": {
            "description": "标签徽章，显示折扣、新品、推荐等标记",
            "use_cases": ["折扣标签", "新品标记", "推荐标签", "健康评级"],
        },
        "List": {
            "description": "列表容器，展示多个商品或属性条目",
            "use_cases": ["商品列表", "营养成分列表", "比较项目列表"],
        },
        "Row": {
            "description": "水平布局容器，横向排列子元素",
            "use_cases": ["价格+单位", "图标+文字", "操作按钮组"],
        },
        "Column": {
            "description": "垂直布局容器，纵向排列子元素",
            "use_cases": ["商品详情", "营养信息", "多行描述"],
        },
        "Icon": {
            "description": "图标元素，表示商品类型、状态、操作等",
            "use_cases": ["商品类型图标", "健康指示", "操作图标"],
        },
    },
    "general": {
        "Card": {
            "description": "通用容器卡片，用于组合展示各类信息",
            "use_cases": ["信息卡片", "通知卡片", "详情展示", "AR标签容器"],
        },
        "Text": {
            "description": "文本元素，显示各类文字内容",
            "use_cases": ["标题文字", "描述内容", "提示信息", "状态文字"],
        },
        "Icon": {
            "description": "图标元素，表示类型、状态、操作等",
            "use_cases": ["类型图标", "状态指示", "操作图标"],
        },
        "Badge": {
            "description": "标签徽章，显示简短的状态或分类信息",
            "use_cases": ["状态标签", "分类标记", "数量徽章"],
        },
        "Row": {
            "description": "水平布局容器，横向排列子元素",
            "use_cases": ["图标+文字", "多元素并排", "操作组"],
        },
        "Column": {
            "description": "垂直布局容器，纵向排列子元素",
            "use_cases": ["多行内容", "信息列表", "分组展示"],
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
