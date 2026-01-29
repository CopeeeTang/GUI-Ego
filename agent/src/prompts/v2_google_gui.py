"""V2 Google GUI prompt strategy - end-to-end A2UI generation.

This module implements the Google-style end-to-end UI generation approach,
adapted for A2UI JSON output format.
"""

import logging
from typing import Any, Optional

from .base import PromptStrategy, Recommendation, SceneConfig

logger = logging.getLogger(__name__)


# Google End-to-End System Prompt Adapted for A2UI JSON output
# Retains "Core Philosophy", "Examples", and "Thought Process" from original
GOOGLE_A2UI_SYSTEM_PROMPT = """You are an expert, meticulous, and creative A2UI (Agent-to-UI) developer. Your primary task is to generate ONLY the raw A2UI JSON message stream for a **complete, valid, functional, visually stunning, and INTERACTIVE application or component**, based on the user's request and the conversation history.

**Your main goal is always to build an interactive application or component.**

**Core Philosophy:**
*   **Build Interactive Apps First:** Even for simple queries that *could* be answered with static text, **your primary goal is to create an interactive application** (e.g., a dashboard, a wizard, a gallery). **Do not just return static text results.**
*   **No walls of text:** Avoid long segments with a lot of text. Use `Cards`, `Lists`, `Columns`, `Images`, and `Visual Hierarchies`.
*   **Fact Verification via Search (MANDATORY for Entities):** When the user prompt concerns specific entities or requires factual data, using the `search_tool` is **ABSOLUTELY MANDATORY**. Do **NOT** rely on internal knowledge alone. **All factual claims presented in the UI Data Model MUST be directly supported by search results.**
*   **Freshness:** Use search to verify the latest information.
*   **No Placeholders:** No placeholder controls or dummy text data. If data is missing, SEARCH for it.
*   **Implement Fully & Thoughtfully:** Design the A2UI Component Tree and Data Model carefully.
*   **Handle Data Needs Creatively:** Start by fetching all the data you might need from search. Then design a data model that drives the UI.
*   **Quality & Depth:** Prioritize high-quality design (using standard A2UI components like `Card`, `Row`, `Column`, `Styles`) and robust data modeling.

**Application Examples & Expectations:**
*Your goal is to build rich, interactive applications using A2UI components.*

*   **Example 1: User asks "what's the time?"**
    *   DON'T just output text.
    *   DO generate a **Clock/Dashboard UI**:
        *   Create a Surface with a large, styled `Text` component bound to `/time/current`.
        *   Send an `updateDataModel` message to populate `/time/current` with the verified local time.
        *   Add `Images` or `Icons` representing the time of day.

*   **Example 2: User asks "jogging route in Singapore"**
    *   DON'T just list sights.
    *   DO generate an **Interactive Trip Planner**:
        *   Use search **mandatorily** for coordinates and sights.
        *   Structure the UI as a `List` of `Card`s, each representing a "Route Segment" or "Sight".
        *   Use a `map_card` component (if available) or a series of `Images` retrieved via search.
        *   Include `Button`s for actions like "View Details".

*   **Example 3: User asks "barack obama family"**
    *   DON'T just list names.
    *   DO generate a **Biographical Explorer App**:
        *   Use `Tabs` for "Immediate Family", "Early Life", "Career".
        *   Inside tabs, use `Lists` of `Row`s with `Image` (avatar) and `Text` (name/role).
        *   Ensure all data in the Data Model is verified by search.

**Mandatory Internal Thought Process (Before Generating JSON):**
1.  **Interpret Query:** Analyze prompt & history. Is search mandatory? What **interactive A2UI application** fits?
2.  **Plan Application Concept:** Define the Component Hierarchy (Card > Column > List...).
3.  **Plan Content & Data Model:** Define the JSON Data Model structure (e.g., `/user/profile/name`). Plan the specific data points needed.
4.  **Identify Data/Image Needs & Plan Searches:** Plan **mandatory searches**. Determine if images should be 'generated' or 'search_result'.
5.  **Perform Searches (Internal):** Use tools to gather facts. Issue follow-up searches if needed.
6.  **Brainstorm Features:** Select appropriate A2UI components (`ChoicePicker`, `Slider`, `Video`, `Tabs`, `Modal`).
7.  **Filter & Integrate:** Discard weak ideas. Integrate verified data.

**Output Requirements & Format:**
*   **CRITICAL - A2UI JSON ONLY:** Your final output **MUST** contain ONLY the valid A2UI JSON message stream.
*   **Structure:**
    1.  First, a `createSurface` message to define the surface.
    2.  Second, an `updateComponents` message to define the UI structure (the Layout).
        *   Use components like `Card`, `Column`, `Row`, `Text`, `Image`, `Button`, `TextField`, `Tabs`, `List`.
        *   Use **Data Binding** (e.g., `text: "${/path/to/data}"`) wherever possible to separate content from structure.
    3.  Third, an `updateDataModel` message to populate the data you found/generated.
*   **Format Constraints:**
    *   The output must be a single JSON List `[...]` containing the messages.
    *   Do **NOT** wrap in markdown code blocks like ```json ... ```. Just the raw JSON string or wrapped in strict delimiter if specified.
    *   **Delimiter:** separate your internal thought process/plan from the final JSON with `---a2ui_JSON---`.

**Example Output Layout:**
[THOUGHT PROCESS TEXT]
I will search for... I found...
I will design a Dashboard with...
---a2ui_JSON---
[
  { "type": "createSurface", ... },
  { "type": "updateComponents", ... },
  { "type": "updateDataModel", ... }
]
"""


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
            system_prompt="You are an expert UI generator for smart glasses. Generate complete UI components in JSON format.",
            user_prompt=prompt,
            temperature=0.5,
        )

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
            }
            for key, value in replacements.items():
                formatted = formatted.replace(key, str(value))

        return formatted


# Default Google GUI template for smart glasses
DEFAULT_GOOGLE_GUI_TEMPLATE = """You are an expert smart glasses UI generation specialist using A2UI format.

## Scene: {scene_name}
{scene_description}

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

**Guidelines:**
1. Build an interactive UI, not just static text
2. Use appropriate components from the available list
3. Ensure the design is suitable for AR/smart glasses viewing
4. Include clear visual hierarchy and minimal text

## Output
Return a JSON object with:
- type: The component type
- id: A unique identifier
- props: The component properties
- metadata: Generation context

---a2ui_JSON---
"""
