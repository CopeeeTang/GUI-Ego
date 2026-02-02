"""V3 Visual prompt strategy - enhanced generation with visual context."""

import json
import logging
import uuid
from typing import Any, Optional

from .base import PromptStrategy, Recommendation, SceneConfig

logger = logging.getLogger(__name__)


# A2UI 原子组件定义 - 与 Preview 渲染器兼容
VISUAL_COMPONENT_DEFINITIONS = {
    "navigation": {
        "Card": {
            "description": "容器卡片，用于组合显示导航信息、地点详情、路线选项",
            "visual_hints": "当需要展示结构化导航信息时使用，可包含多个子元素",
            "anchor_strategy": "通常显示在视野下方或角落",
        },
        "Text": {
            "description": "文本元素，显示标题、描述、距离、时间等",
            "visual_hints": "用于显示地点名称、距离、时间等文字信息",
            "anchor_strategy": "作为Card或其他容器的子元素",
        },
        "Button": {
            "description": "交互按钮，触发导航、选择路线等操作",
            "visual_hints": "当用户需要进行操作选择时使用",
            "anchor_strategy": "通常放置在卡片底部或右侧",
        },
        "Icon": {
            "description": "图标元素，表示方向、地点类型、交通方式等",
            "visual_hints": "用于增强视觉识别，配合文字使用",
            "anchor_strategy": "放置在文字左侧或独立显示",
        },
        "Badge": {
            "description": "标签徽章，显示距离、时间等简短信息",
            "visual_hints": "用于突出显示关键数值或状态",
            "anchor_strategy": "放置在相关内容旁边",
        },
        "Row": {
            "description": "水平布局容器，横向排列子元素",
            "visual_hints": "用于并排显示多个元素",
            "anchor_strategy": "作为布局容器使用",
        },
        "Column": {
            "description": "垂直布局容器，纵向排列子元素",
            "visual_hints": "用于垂直排列多行内容",
            "anchor_strategy": "作为布局容器使用",
        },
    },
    "shopping": {
        "Card": {
            "description": "容器卡片，用于显示商品信息、价格比较、营养数据",
            "visual_hints": "当视野中有商品或价格标签时使用",
            "anchor_strategy": "锚定到产品附近或视野中央",
        },
        "Text": {
            "description": "文本元素，显示商品名称、价格、营养成分等",
            "visual_hints": "用于显示商品相关的文字信息",
            "anchor_strategy": "作为Card的子元素",
        },
        "Button": {
            "description": "交互按钮，添加购物车、查看详情等",
            "visual_hints": "当用户需要进行购物操作时使用",
            "anchor_strategy": "放置在卡片底部",
        },
        "Badge": {
            "description": "标签徽章，显示折扣、健康评级等",
            "visual_hints": "用于突出显示促销信息或评级",
            "anchor_strategy": "放置在商品信息旁边",
        },
        "List": {
            "description": "列表容器，展示多个商品或属性条目",
            "visual_hints": "当需要展示多个可比较项目时使用",
            "anchor_strategy": "作为Card的子元素",
        },
        "Row": {
            "description": "水平布局容器，横向排列子元素",
            "visual_hints": "用于并排显示价格和单位等",
            "anchor_strategy": "作为布局容器使用",
        },
        "Column": {
            "description": "垂直布局容器，纵向排列子元素",
            "visual_hints": "用于垂直排列营养成分等",
            "anchor_strategy": "作为布局容器使用",
        },
        "Icon": {
            "description": "图标元素，表示商品类型、健康指示等",
            "visual_hints": "用于增强视觉识别",
            "anchor_strategy": "放置在文字左侧",
        },
    },
}

# A2UI 原子组件 Schema 定义
COMPONENT_SCHEMAS = {
    "Card": {
        "required": ["children"],
        "properties": {
            "variant": "string - 卡片变体 (default|glass|outlined)",
            "children": "array - 子组件列表",
            "padding": "string - 内边距 (sm|md|lg)",
        },
    },
    "Text": {
        "required": ["content"],
        "properties": {
            "content": "string - 文本内容",
            "variant": "string - 文本变体 (title|subtitle|body|caption)",
            "color": "string - 文本颜色",
        },
    },
    "Button": {
        "required": ["label"],
        "properties": {
            "label": "string - 按钮文字",
            "variant": "string - 按钮变体 (primary|secondary|ghost)",
            "icon": "string - 图标名称（可选）",
            "action": "object - 操作定义 {type, payload}",
        },
    },
    "Icon": {
        "required": ["name"],
        "properties": {
            "name": "string - 图标名称 (arrow-right|location|cart|star|...)",
            "size": "string - 图标大小 (sm|md|lg)",
            "color": "string - 图标颜色",
        },
    },
    "Badge": {
        "required": ["text"],
        "properties": {
            "text": "string - 徽章文字",
            "variant": "string - 徽章变体 (default|success|warning|error)",
        },
    },
    "Row": {
        "required": ["children"],
        "properties": {
            "children": "array - 子组件列表",
            "gap": "string - 间距 (sm|md|lg)",
            "align": "string - 对齐方式 (start|center|end|between)",
        },
    },
    "Column": {
        "required": ["children"],
        "properties": {
            "children": "array - 子组件列表",
            "gap": "string - 间距 (sm|md|lg)",
        },
    },
    "List": {
        "required": ["items"],
        "properties": {
            "items": "array - 列表项",
            "variant": "string - 列表变体 (default|compact)",
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
    VISUAL_PROMPT_TEMPLATE = """你是一个智能眼镜 UI 生成专家，能够结合视觉场景信息和用户个人信息生成最合适的 UI 组件。

## 当前场景: {scene_name}

## 用户画像
{user_profile}

## 当前活动上下文
{scene_context}

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
结合用户画像、活动上下文、视觉上下文和推荐内容，生成最合适的 UI 组件。

考虑以下因素：
1. **用户个性化**: 根据用户的兴趣、性格和目标定制 UI 内容和交互方式
2. **物体定位**: 根据视觉上下文中识别的物体位置，确定 AR 元素的锚点
3. **环境适配**: 根据环境特征（光线、拥挤程度）调整 UI 显示策略
4. **用户活动**: 根据用户当前活动状态选择合适的交互方式
5. **视觉一致性**: 确保 UI 元素与物理场景自然融合

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

        # Build user profile context string
        user_profile_str = "无用户画像信息"
        metadata = getattr(recommendation, 'metadata', {}) or {}
        user_profile = metadata.get("user_profile", {})
        if user_profile:
            profile_parts = []
            if user_profile.get("preferred_name"):
                profile_parts.append(f"姓名: {user_profile['preferred_name']}")
            if user_profile.get("occupation"):
                profile_parts.append(f"职业: {user_profile['occupation']}")
            if user_profile.get("personality"):
                profile_parts.append(f"性格: {user_profile['personality']}")
            if user_profile.get("interests"):
                interests = ", ".join(user_profile["interests"][:5])
                profile_parts.append(f"兴趣: {interests}")
            if user_profile.get("goals"):
                profile_parts.append(f"当前目标: {user_profile['goals']}")
            if profile_parts:
                user_profile_str = "\n".join(profile_parts)

        # Build scene context string
        scene_context_str = metadata.get("scene_context", "") or "无活动上下文信息"

        return self.VISUAL_PROMPT_TEMPLATE.format(
            scene_name=scene.name,
            visual_context=visual_context_str,
            recommendation_type=recommendation.type,
            recommendation_content=recommendation.content,
            component_list="\n".join(component_lines),
            component_schema="\n\n".join(schema_lines),
            user_profile=user_profile_str,
            scene_context=scene_context_str,
        )

    def _infer_component_type(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
    ) -> str:
        """Infer component type from recommendation if not provided by LLM.

        Returns A2UI atomic component types.
        """
        # For A2UI atomic components, Card is the primary container
        # Most UI patterns start with a Card containing other elements
        return "Card"
