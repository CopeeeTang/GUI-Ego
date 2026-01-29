"""V3 Visual prompt strategy - enhanced generation with visual context."""

import json
import logging
import uuid
from typing import Any, Optional

from .base import PromptStrategy, Recommendation, SceneConfig

logger = logging.getLogger(__name__)


# Component definitions with visual context considerations
VISUAL_COMPONENT_DEFINITIONS = {
    "navigation": {
        "map_card": {
            "description": "地图卡片，用于显示位置、导航路线、附近的兴趣点",
            "visual_hints": "当视野中有开阔道路或户外环境时特别适用",
            "anchor_strategy": "通常显示在视野下方或角落",
        },
        "ar_label": {
            "description": "AR 标签，悬浮在物理对象上方显示信息",
            "visual_hints": "当视野中有明确的建筑物、地标或物体时使用",
            "anchor_strategy": "锚定到识别出的物体上方",
        },
        "direction_arrow": {
            "description": "方向指引箭头，显示前进方向和距离",
            "visual_hints": "当用户正在移动或需要导航指引时使用",
            "anchor_strategy": "显示在视野中央或前进方向",
        },
        "comparison_card": {
            "description": "对比卡片，并排比较两个或多个选项",
            "visual_hints": "当需要在多个选择间决策时使用",
            "anchor_strategy": "通常显示在视野中央",
        },
    },
    "shopping": {
        "comparison_card": {
            "description": "对比卡片，并排比较商品属性",
            "visual_hints": "当视野中有多个可比较的商品时使用",
            "anchor_strategy": "显示在商品上方或视野中央",
        },
        "nutrition_card": {
            "description": "营养信息卡片，显示食品的营养数据",
            "visual_hints": "当视野中有食品包装或产品时使用",
            "anchor_strategy": "锚定到产品包装附近",
        },
        "price_calculator": {
            "description": "价格计算器，计算单价、总价、性价比",
            "visual_hints": "当视野中有价格标签或多个商品时使用",
            "anchor_strategy": "显示在价格标签附近",
        },
        "ar_label": {
            "description": "AR 标签，叠加在商品上显示信息",
            "visual_hints": "当需要标注特定商品时使用",
            "anchor_strategy": "锚定到商品位置",
        },
    },
}

# Component schemas for props generation
COMPONENT_SCHEMAS = {
    "map_card": {
        "required": ["title"],
        "properties": {
            "title": "string - 卡片标题",
            "subtitle": "string - 副标题（可选）",
            "markers": "array - 地图标记点列表",
            "action": "object - 操作按钮 {label, type}",
        },
    },
    "ar_label": {
        "required": ["text"],
        "properties": {
            "text": "string - 主要文字",
            "subtext": "string - 次要文字（可选）",
            "icon": "string - 图标名称（可选）",
            "anchor": "object - 锚定信息 {type, target, position}",
        },
    },
    "comparison_card": {
        "required": ["title", "items"],
        "properties": {
            "title": "string - 对比标题",
            "items": "array - 对比项列表",
            "recommendation": "string - 推荐结论",
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
    },
    "nutrition_card": {
        "required": ["product_name"],
        "properties": {
            "product_name": "string - 产品名称",
            "calories": "number - 热量(kcal)",
            "protein": "string - 蛋白质",
            "carbs": "string - 碳水化合物",
            "fat": "string - 脂肪",
            "health_rating": "integer - 健康评级 1-5",
        },
    },
    "price_calculator": {
        "required": ["items"],
        "properties": {
            "title": "string - 标题",
            "items": "array - 商品列表",
            "total": "number - 总价",
            "recommendation": "string - 推荐建议",
        },
    },
}


class VisualPromptStrategy(PromptStrategy):
    """Visual-enhanced prompt strategy using video frame context.

    This strategy uses visual context from video frames to enhance
    UI component generation. It considers:
    - Scene environment and objects visible in the frames
    - Suitable anchor points for AR elements
    - User activity and context

    Supports two visual context modes:
    - DIRECT: Pass images directly to multimodal LLM
    - DESCRIPTION: Use pre-generated text descriptions
    """

    # Main prompt template with visual context
    VISUAL_PROMPT_TEMPLATE = """你是一个智能眼镜 UI 生成专家，能够结合视觉场景信息生成最合适的 UI 组件。

## 当前场景: {scene_name}

## 视觉上下文
{visual_context}

## AI 推荐
- 类型: {recommendation_type}
- 内容: {recommendation_content}

## 可用组件
{component_list}

## 组件 Schema
{component_schema}

## 任务
结合视觉上下文和推荐内容，生成最合适的 UI 组件。

考虑以下因素：
1. **物体定位**: 根据视觉上下文中识别的物体位置，确定 AR 元素的锚点
2. **环境适配**: 根据环境特征（光线、拥挤程度）调整 UI 显示策略
3. **用户活动**: 根据用户当前活动状态选择合适的交互方式
4. **视觉一致性**: 确保 UI 元素与物理场景自然融合

## 输出格式
返回完整的 JSON 组件定义:
{{
    "type": "组件类型",
    "id": "唯一标识符",
    "props": {{
        // 组件属性
    }},
    "visual_anchor": {{
        "type": "object|location|screen",
        "target": "锚定目标描述",
        "position": "left|center|right|top|bottom",
        "reasoning": "为什么选择这个锚点"
    }}
}}
"""

    @property
    def name(self) -> str:
        return "v3_with_visual"

    @property
    def description(self) -> str:
        return "Visual-enhanced generation using video frame context"

    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize the visual prompt strategy.

        Args:
            llm_client: LLM client instance with multimodal support.
        """
        self._llm_client = llm_client

    def set_llm_client(self, llm_client: Any) -> None:
        """Set the LLM client."""
        self._llm_client = llm_client

    def supports_visual_context(self) -> bool:
        """This strategy is designed for visual context."""
        return True

    def generate(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
        visual_context: Optional[dict] = None,
    ) -> dict:
        """Generate UI component with visual context enhancement.

        Args:
            recommendation: The AI recommendation to convert to UI.
            scene: Scene configuration with allowed components.
            visual_context: Visual context from video frames (required).

        Returns:
            Dictionary containing the generated UI component with visual anchor info.

        Raises:
            ValueError: If visual context is not provided.
        """
        if self._llm_client is None:
            raise RuntimeError("LLM client not set. Call set_llm_client first.")

        self.validate_inputs(recommendation, scene)

        if visual_context is None:
            logger.warning(
                "Visual context not provided for v3_with_visual strategy. "
                "Falling back to description-only mode."
            )
            visual_context = {"mode": "description", "description": "No visual context available."}

        # Build the prompt
        prompt = self._build_prompt(recommendation, scene, visual_context)

        # Check if we should use multimodal or text-only
        if visual_context.get("mode") == "direct" and visual_context.get("frames_base64"):
            # Use multimodal completion
            logger.info(f"Using multimodal generation with {len(visual_context['frames_base64'])} frames")
            result = self._llm_client.complete_json_with_images(
                system_prompt="你是一个智能眼镜 UI 生成专家。结合视觉信息生成 JSON 格式的 UI 组件。",
                user_prompt=prompt,
                images=visual_context["frames_base64"],
                temperature=0.5,
            )
        else:
            # Use text-only completion
            logger.info("Using text-only generation with visual description")
            result = self._llm_client.complete_json(
                system_prompt="你是一个智能眼镜 UI 生成专家。生成 JSON 格式的 UI 组件。",
                user_prompt=prompt,
                temperature=0.5,
            )

        # Ensure required fields
        if "type" not in result:
            result["type"] = self._infer_component_type(recommendation, scene)
        if "id" not in result:
            result["id"] = f"{result['type']}_{uuid.uuid4().hex[:8]}"

        # Add metadata
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["selection"] = {
            "strategy": self.name,
            "visual_context_mode": visual_context.get("mode"),
            "has_visual_anchor": "visual_anchor" in result,
        }
        result["metadata"]["source_recommendation"] = {
            "type": recommendation.type,
            "content": recommendation.content,
            "time_range": [recommendation.start_time, recommendation.end_time],
        }

        return result

    def _build_prompt(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
        visual_context: dict,
    ) -> str:
        """Build the complete prompt with visual context."""
        # Get scene-specific component definitions
        scene_components = VISUAL_COMPONENT_DEFINITIONS.get(scene.name, {})

        # Build component list with visual hints
        component_lines = []
        for comp_name in scene.allowed_components:
            comp_info = scene_components.get(comp_name, {})
            component_lines.append(
                f"- **{comp_name}**: {comp_info.get('description', 'No description')}\n"
                f"  视觉提示: {comp_info.get('visual_hints', 'N/A')}\n"
                f"  锚定策略: {comp_info.get('anchor_strategy', 'N/A')}"
            )

        # Build component schema section
        schema_lines = []
        for comp_name in scene.allowed_components:
            schema = COMPONENT_SCHEMAS.get(comp_name, {})
            if schema:
                props = "\n    ".join(f"- {k}: {v}" for k, v in schema.get("properties", {}).items())
                schema_lines.append(
                    f"### {comp_name}\n"
                    f"  必需字段: {', '.join(schema.get('required', []))}\n"
                    f"  属性:\n    {props}"
                )

        # Format visual context
        if visual_context.get("mode") == "direct":
            visual_context_str = (
                "[视觉上下文: 图像已直接传递给模型]\n"
                "请仔细分析图像中的:\n"
                "1. 环境类型和场所\n"
                "2. 主要物体及其位置\n"
                "3. 可见的文字或标志\n"
                "4. 用户可能的活动"
            )
        else:
            visual_context_str = visual_context.get("description", "无视觉上下文")

        return self.VISUAL_PROMPT_TEMPLATE.format(
            scene_name=scene.name,
            visual_context=visual_context_str,
            recommendation_type=recommendation.type,
            recommendation_content=recommendation.content,
            component_list="\n".join(component_lines),
            component_schema="\n\n".join(schema_lines),
        )

    def _infer_component_type(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
    ) -> str:
        """Infer component type from recommendation if not provided by LLM."""
        # Simple heuristic based on recommendation type and content
        content_lower = recommendation.content.lower()

        if scene.name == "navigation":
            if any(kw in content_lower for kw in ["导航", "路线", "地图", "附近"]):
                return "map_card"
            elif any(kw in content_lower for kw in ["建筑", "地标", "识别"]):
                return "ar_label"
            elif any(kw in content_lower for kw in ["方向", "转", "前进"]):
                return "direction_arrow"
            else:
                return "ar_label"  # Default for navigation
        elif scene.name == "shopping":
            if any(kw in content_lower for kw in ["比较", "对比", "选择"]):
                return "comparison_card"
            elif any(kw in content_lower for kw in ["营养", "热量", "成分"]):
                return "nutrition_card"
            elif any(kw in content_lower for kw in ["价格", "性价比", "计算"]):
                return "price_calculator"
            else:
                return "ar_label"  # Default for shopping

        # Fallback to first allowed component
        return scene.allowed_components[0] if scene.allowed_components else "ar_label"
