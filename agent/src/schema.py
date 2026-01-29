"""
Unified schema definitions for the Generative UI Pipeline.

This module provides the single source of truth for all data models
used across the pipeline, eliminating duplication between data_loader
and prompts modules.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Recommendation:
    """AI recommendation data structure.

    This unified class supports both:
    - Loading from annotation JSON (data_loader)
    - Prompt generation (prompts module)

    Attributes:
        id: Unique identifier for the recommendation.
        type: Type of recommendation (e.g., "navigation", "product_info").
        content: The recommendation content/text.
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        confidence: Confidence score (0-1).
        is_accepted: Whether the recommendation was accepted (from annotation).
        object_list: List of related objects (from annotation).
        metadata: Additional recommendation metadata.
    """

    id: str
    type: str
    content: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None
    is_accepted: Optional[bool] = None
    object_list: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def from_annotation(cls, annotation: dict, rec_id: str = "") -> "Recommendation":
        """Create a Recommendation from an annotation dictionary."""
        # Handle time_interval format from annotations
        time_interval = annotation.get("time_interval", {})
        start_time = time_interval.get("start", annotation.get("start_time", 0))
        end_time = time_interval.get("end", annotation.get("end_time", 0))

        return cls(
            id=rec_id or annotation.get("id", ""),
            type=annotation.get("type", "general"),
            content=annotation.get("text", annotation.get("content", "")),
            start_time=start_time,
            end_time=end_time,
            confidence=annotation.get("confidence"),
            is_accepted=annotation.get("is_accepted"),
            object_list=annotation.get("object_list"),
            metadata=annotation.get("metadata"),
        )


@dataclass
class SceneConfig:
    """Configuration for a scene context.

    Attributes:
        name: Scene name (e.g., "navigation", "shopping").
        allowed_components: List of component types allowed in this scene.
        start_time: Scene start timestamp (optional, for filtering).
        end_time: Scene end timestamp (optional, for filtering).
        description: Human-readable scene description.
        metadata: Additional scene-specific configuration.
    """

    name: str
    allowed_components: list[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    description: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


# Unified component registry with all fields
COMPONENT_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "navigation": {
        "map_card": {
            "description": "地图卡片，用于显示位置、导航路线、附近的兴趣点（如共享单车、建筑）",
            "use_cases": ["找共享单车", "导航到目的地", "显示附近设施", "估算距离/时间"],
            "visual_hints": "当视野中有开阔道路或户外环境时特别适用",
            "anchor_strategy": "通常显示在视野下方或角落",
        },
        "ar_label": {
            "description": "AR 标签，悬浮在物理对象上方显示信息",
            "use_cases": ["识别建筑名称", "标注地标", "显示方向指示", "物体识别结果"],
            "visual_hints": "当视野中有明确的建筑物、地标或物体时使用",
            "anchor_strategy": "锚定到识别出的物体上方",
        },
        "direction_arrow": {
            "description": "方向指引箭头，显示前进方向和距离",
            "use_cases": ["步行导航", "指示转弯方向", "显示目的地距离"],
            "visual_hints": "当用户正在移动或需要导航指引时使用",
            "anchor_strategy": "显示在视野中央或前进方向",
        },
        "comparison_card": {
            "description": "对比卡片，并排比较两个或多个选项",
            "use_cases": ["比较步行vs骑车时间", "比较路线选项", "决策支持"],
            "visual_hints": "当需要在多个选择间决策时使用",
            "anchor_strategy": "通常显示在视野中央",
        },
    },
    "shopping": {
        "comparison_card": {
            "description": "对比卡片，并排比较商品属性",
            "use_cases": ["比较商品新鲜度", "比较价格", "比较营养成分", "决策支持"],
            "visual_hints": "当视野中有多个可比较的商品时使用",
            "anchor_strategy": "显示在商品上方或视野中央",
        },
        "nutrition_card": {
            "description": "营养信息卡片，显示食品的营养数据",
            "use_cases": ["显示热量", "显示蛋白质/碳水/脂肪", "显示成分列表", "健康评级"],
            "visual_hints": "当视野中有食品包装或产品时使用",
            "anchor_strategy": "锚定到产品包装附近",
        },
        "price_calculator": {
            "description": "价格计算器，计算单价、总价、性价比",
            "use_cases": ["计算单价", "比较性价比", "估算总价"],
            "visual_hints": "当视野中有价格标签或多个商品时使用",
            "anchor_strategy": "显示在价格标签附近",
        },
        "ar_label": {
            "description": "AR 标签，叠加在商品上显示信息",
            "use_cases": ["显示商品名称", "显示价格", "标注推荐商品"],
            "visual_hints": "当需要标注特定商品时使用",
            "anchor_strategy": "锚定到商品位置",
        },
    },
}


def get_component_definitions(scene: str, include_visual: bool = False) -> dict[str, dict]:
    """Get component definitions for a scene.

    Args:
        scene: Scene name.
        include_visual: Whether to include visual_hints and anchor_strategy.

    Returns:
        Dictionary of component definitions.
    """
    scene_components = COMPONENT_REGISTRY.get(scene, {})

    if include_visual:
        return scene_components

    # Return without visual fields for backward compatibility
    return {
        name: {
            "description": info["description"],
            "use_cases": info["use_cases"],
        }
        for name, info in scene_components.items()
    }


def get_allowed_components(scene: str) -> list[str]:
    """Get list of allowed component types for a scene."""
    return list(COMPONENT_REGISTRY.get(scene, {}).keys())
