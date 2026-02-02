"""V2 Google GUI prompt strategy - end-to-end A2UI generation.

This module implements the Google-style end-to-end UI generation approach,
adapted for A2UI JSON output format.
"""

import logging
from typing import Any, Optional

from .base import PromptStrategy, Recommendation, SceneConfig

logger = logging.getLogger(__name__)


# Google End-to-End System Prompt for A2UI component generation
# Simplified to output direct A2UI component format (compatible with converter.py)
GOOGLE_A2UI_SYSTEM_PROMPT = """You are an expert A2UI (Agent-to-UI) developer for smart glasses AR interfaces.
Your task is to generate a single, complete A2UI component as a JSON object.

**Core Philosophy:**
- **Build Interactive UIs:** Create visually engaging components, not just static text
- **Minimal & Clear:** Smart glasses have limited display - use visual hierarchy wisely
- **No Placeholders:** Always include real, meaningful content based on the context
- **AR-Optimized:** Design for floating overlays, quick glances, and ambient awareness
- **Semantic Only:** Output semantic structure, NOT styling. The renderer handles all visuals.

**Available A2UI Components with Variants:**

| Component   | Variants                              | Usage                           |
|-------------|---------------------------------------|--------------------------------|
| Card        | glass, solid, outline, alert          | Container for UI sections       |
| Text        | hero, h1, h2, body, caption, label    | Text display                    |
| Button      | primary, secondary, ghost, icon_only  | Interactive actions             |
| Badge       | info, success, warning, error         | Status indicators               |
| Icon        | small, medium, large                  | Material icons                  |
| ProgressBar | default, slim                         | Progress indicators             |
| Row         | (layout only)                         | Horizontal layout               |
| Column      | (layout only)                         | Vertical layout                 |
| List        | (layout only)                         | List container                  |
| Divider     | (layout only)                         | Visual separator                |
| Image       | (layout only)                         | Image display                   |

**Output Format:**
Return a single JSON object with this structure:
{
  "type": "<component_type>",
  "id": "<unique_identifier>",
  "props": {
    "variant": "<variant_token>",  // Use variant tokens, NOT raw styles
    ...semantic props only
  },
  "children": [ ... ],
  "metadata": {
    "selection": {
      "strategy": "v2_google_gui",
      "end_to_end": true
    }
  }
}

**FORBIDDEN - Never include these in output:**
- "style" objects at any level
- "color", "fontSize", "fontWeight", "background" in props
- Any raw CSS values or inline styles
- Hex colors, pixel values, or CSS properties

**Design Guidelines for Smart Glasses:**
1. Use Card with variant="glass" as the root container
2. Prefer Row > [Icon, Column > [Text(variant="h2"), Text(variant="caption")]] for info labels
3. Keep text concise - no long paragraphs
4. Use icons to convey meaning quickly
5. Use Badge with appropriate variant for status indicators
6. Include actionable buttons when interaction is expected

**CRITICAL:** Output ONLY the raw JSON object. No markdown, no code blocks, no explanations."""


class GoogleGUIPromptStrategy(PromptStrategy):
    """Google GUI end-to-end generation strategy.

    This strategy uses a single prompt to generate the complete UI component
    end-to-end without the two-step selection + filling process. It follows
    Google's approach of building interactive applications rather than static
    text responses.
    """

    @property
    def name(self) -> str:
        return "v2_google_gui"

    @property
    def description(self) -> str:
        return "Google GUI end-to-end generation (requires custom prompt template)"

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        prompt_template_path: Optional[str] = None,
        use_default_template: bool = True,
    ):
        """Initialize the Google GUI strategy.

        Args:
            llm_client: LLM client instance.
            prompt_template: The prompt template string.
            prompt_template_path: Path to a file containing the prompt template.
            use_default_template: If True and no template provided, use DEFAULT_GOOGLE_GUI_TEMPLATE.
        """
        self._llm_client = llm_client
        self._prompt_template = prompt_template

        # Load template from file if path provided
        if prompt_template_path and not prompt_template:
            self._load_template_from_file(prompt_template_path)

        # Use default template if none provided
        if self._prompt_template is None and use_default_template:
            self._prompt_template = DEFAULT_GOOGLE_GUI_TEMPLATE
            logger.info("Using default Google GUI template")

    def _load_template_from_file(self, path: str) -> None:
        """Load prompt template from file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._prompt_template = f.read()
            logger.info(f"Loaded prompt template from: {path}")
        except Exception as e:
            logger.error(f"Failed to load prompt template from {path}: {e}")
            raise

    def set_llm_client(self, llm_client: Any) -> None:
        """Set the LLM client."""
        self._llm_client = llm_client

    def set_prompt_template(self, template: str) -> None:
        """Set the prompt template."""
        self._prompt_template = template

    def _strip_style_properties(self, component: dict) -> dict:
        """Remove forbidden style properties from component tree.

        This sanitizer ensures the LLM output conforms to the semantic-only
        constraint by stripping any CSS/styling properties that may have leaked.

        Args:
            component: A2UI component dictionary (potentially with children)

        Returns:
            Sanitized component with all style properties removed
        """
        # Remove top-level style object
        if "style" in component:
            del component["style"]

        # Remove forbidden properties from props
        if "props" in component:
            forbidden_props = [
                "color", "fontSize", "fontWeight", "background",
                "backgroundColor", "borderColor", "textColor",
                "padding", "margin", "width", "height",
            ]
            for forbidden in forbidden_props:
                component["props"].pop(forbidden, None)

        # Recursively sanitize children
        if "children" in component and isinstance(component["children"], list):
            component["children"] = [
                self._strip_style_properties(c) if isinstance(c, dict) else c
                for c in component["children"]
            ]

        # Also handle "content" field if it contains nested components
        if "content" in component:
            content = component["content"]
            if isinstance(content, dict):
                component["content"] = self._strip_style_properties(content)
            elif isinstance(content, list):
                component["content"] = [
                    self._strip_style_properties(c) if isinstance(c, dict) else c
                    for c in content
                ]

        return component

    def supports_visual_context(self) -> bool:
        """Google GUI strategy can support visual context if template includes it."""
        return True  # Depends on the template

    def generate(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
        visual_context: Optional[dict] = None,
    ) -> dict:
        """Generate UI component using end-to-end approach.

        Args:
            recommendation: The AI recommendation to convert to UI.
            scene: Scene configuration with allowed components.
            visual_context: Optional visual context from video frames.

        Returns:
            Dictionary containing the generated UI component.

        Raises:
            NotImplementedError: If prompt template is not provided.
        """
        if self._prompt_template is None:
            raise NotImplementedError(
                "Google GUI prompt template not provided. "
                "Please set a prompt template using set_prompt_template() or "
                "provide prompt_template_path in constructor."
            )

        if self._llm_client is None:
            raise RuntimeError("LLM client not set. Call set_llm_client first.")

        self.validate_inputs(recommendation, scene)

        # Format the prompt template
        prompt = self._format_prompt(recommendation, scene, visual_context)

        # Call LLM for end-to-end generation
        logger.info(f"Generating component using Google GUI approach for recommendation {recommendation.id}")

        result = self._llm_client.complete_json(
            system_prompt=GOOGLE_A2UI_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.5,
        )

        # Sanitize output to ensure semantic-only structure
        result = self._strip_style_properties(result)

        # Add metadata
        result["metadata"] = result.get("metadata", {})
        result["metadata"]["selection"] = {
            "strategy": self.name,
            "end_to_end": True,
        }

        return result

    def _format_prompt(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
        visual_context: Optional[dict] = None,
    ) -> str:
        """Format the prompt template with provided data."""
        # Build component list
        component_list = ", ".join(scene.allowed_components)

        # Build visual context string
        visual_context_str = ""
        if visual_context:
            if visual_context.get("mode") == "description":
                visual_context_str = visual_context.get("description", "")
            elif visual_context.get("mode") == "direct":
                visual_context_str = "[Visual context: Images passed directly]"

        # Build user profile context string
        user_profile_str = ""
        metadata = getattr(recommendation, 'metadata', {}) or {}
        user_profile = metadata.get("user_profile", {})
        if user_profile:
            profile_parts = []
            if user_profile.get("preferred_name"):
                profile_parts.append(f"Name: {user_profile['preferred_name']}")
            if user_profile.get("occupation"):
                profile_parts.append(f"Occupation: {user_profile['occupation']}")
            if user_profile.get("personality"):
                profile_parts.append(f"Personality: {user_profile['personality']}")
            if user_profile.get("interests"):
                interests = ", ".join(user_profile["interests"][:5])  # Limit to 5
                profile_parts.append(f"Interests: {interests}")
            if user_profile.get("goals"):
                profile_parts.append(f"Current Goals: {user_profile['goals']}")
            user_profile_str = "\n".join(profile_parts)

        # Build scene context string
        scene_context_str = metadata.get("scene_context", "") or ""

        # Format template
        try:
            formatted = self._prompt_template.format(
                scene_name=scene.name,
                scene_description=scene.description or "",
                component_list=component_list,
                recommendation_id=recommendation.id,
                recommendation_type=recommendation.type,
                recommendation_content=recommendation.content,
                visual_context=visual_context_str,
                user_profile=user_profile_str,
                scene_context=scene_context_str,
            )
        except KeyError as e:
            logger.warning(f"Template variable not found: {e}. Using partial formatting.")
            # Fallback to partial formatting
            formatted = self._prompt_template
            replacements = {
                "{scene_name}": scene.name,
                "{scene_description}": scene.description or "",
                "{component_list}": component_list,
                "{recommendation_id}": recommendation.id,
                "{recommendation_type}": recommendation.type,
                "{recommendation_content}": recommendation.content,
                "{visual_context}": visual_context_str,
                "{user_profile}": user_profile_str,
                "{scene_context}": scene_context_str,
            }
            for key, value in replacements.items():
                formatted = formatted.replace(key, str(value))

        return formatted


# Default Google GUI template for smart glasses
DEFAULT_GOOGLE_GUI_TEMPLATE = """You are an expert smart glasses UI generation specialist using A2UI format.

## Scene: {scene_name}
{scene_description}

## User Profile
{user_profile}

## Current Activity Context
{scene_context}

## Available Components
{component_list}

## Visual Context
{visual_context}

## AI Recommendation
- ID: {recommendation_id}
- Type: {recommendation_type}
- Content: {recommendation_content}

## Task
Generate a complete, interactive A2UI component that best represents this recommendation for smart glasses display.
Consider the user's profile, interests, and current activity context when designing the UI.

**Guidelines:**
1. Build an interactive UI, not just static text
2. Use appropriate components from the available list
3. Ensure the design is suitable for AR/smart glasses viewing
4. Include clear visual hierarchy and minimal text
5. Personalize content based on user profile and context

## Output
Return a JSON object with:
- type: The component type
- id: A unique identifier
- props: The component properties
- metadata: Generation context

---a2ui_JSON---
"""
