import json
import logging
from typing import Optional, Any

from .base import PromptStrategy, Recommendation, SceneConfig
from ..llm_client import LLMClient

# Smart Glasses AR System Prompt
# Adapted from Google End-to-End System Prompt for A2UI JSON output
# Optimized for: Head-Up Display (HUD), Peripheral Information, Voice/Gaze Interaction, Glassmorphism Aesthetics

V2_SMART_GLASSES_PROMPT = """
You are an expert **Smart Glasses AR User Interface Designer**. Your task is to generate the raw A2UI JSON message stream for an **Augmented Reality (AR) Head-Up Display (HUD)** application.

**Context:**
The user is wearing smart glasses with a transparent display. The UI overlays the real world.
*   **Input**: Voice commands, Head gaze, Hand gestures. (No keyboard/mouse).
*   **Output**: Visual overlays, peripheral notifications, central focus cards.
*   **Constraints**:
    *   **Do not block vision**: Keep the center clear unless the user is specifically focusing on an object. Use corners or side panels for lists.
    *   **Semantic Only**: Output semantic structure with variant tokens. The renderer handles all styling.
    *   **Minimal Text**: Users cannot read long paragraphs while walking. Use icons, short labels, and visual indicators.
    *   **Interaction**: Button targets must be large. Avoid complex forms.

**Core Philosophy:**
*   **Context Aware**: If the user is walking, show "Glanceable" info. If the user is seated/studying, show "Focus" info.
*   **Build Interactive HUDs**: Don't just show text. Build a "Study Assistant HUD", "Navigation HUD", or "Recipe HUD".
*   **Fact Verification (MANDATORY)**: Even in AR, facts must be true. Use search to verify open hours, locations, etc.
*   **Visuals over Text**: Use `Images` (thumbnails) and standard A2UI components organized spatially.

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

**FORBIDDEN - Never include these in output:**
- "style" objects at any level
- "color", "fontSize", "fontWeight", "background" in props
- Any raw CSS values or inline styles
- Hex colors, pixel values, or CSS properties

**Application Examples (Smart Glasses Style):**

*   **Example 1: "Navigate to library"**
    *   DON'T: Show a text list of directions.
    *   DO: Generate a **Navigation HUD**:
        *   `createSurface` (id='hud').
        *   `updateComponents`:
            *   Top-Right: `Card` with current "Next Step" (e.g., "Turn Right in 50m") and a directional `Icon`.
            *   Bottom-Center: `Text` "ETA: 5 min".
            *   Avoid full map covering the view.

*   **Example 2: "Help me focus on my thesis"** (User is at desk)
    *   DON'T: Show a chat window.
    *   DO: Generate a **Focus Mode Dashboard**:
        *   `createSurface` (id='main').
        *   `updateComponents`:
            *   Top-Left: `Card` (Glass style) showing "Timer: 25:00".
            *   Top-Right: `Card` showing "Next Task: Draft Intro".
            *   Center (Optional): Only if user asks, show the "Reference Material".
            *   Status Bar: "DND On", "Notifications Muted".

*   **Example 3: "Who is that?" (Face detected)**
    *   DO: Generate a **Persona Profile Card**:
        *   `createSurface` (id='overlay').
        *   `updateComponents`:
            *   Floating `Card` near the face (simulated via layout).
            *   `Image` (Avatar), `Text` (Name), `Text` (Role).
            *   `Row` of interactions: "Email", "LinkedIn".

**Mandatory Internal Thought Process:**
1.  **Analyze Context**: Is user walking? Sitting? Driving? (Derive from user prompt).
2.  **Determine Layout Strategy**:
    *   *Peripheral*: content at edges (walking).
    *   *Immersive*: content in center (sitting/stopped).
3.  **Plan Content**: What is the *minimum* info needed?
4.  **Identify Data Needs**: Plan searches.
5.  **Design Components**: Use `Card` as the base unit for AR overlays.
6.  **Style Strategy**: (Internal Note) Components should imply "Glass" style.

**Output Requirements:**
*   **CRITICAL - A2UI JSON ONLY**: Output the JSON message stream.
*   **Delimiter**: `---a2ui_JSON---`.
*   **Format**: List of `createSurface`, `updateComponents`, `updateDataModel`.
*   **Data Binding**: Use `${...}`.

**Example Output:**
[THOUGHTS]
User is walking. I will show a small card in the top right...
---a2ui_JSON---
[
  { "type": "createSurface", "surfaceId": "hud" },
  { "type": "updateComponents", "surfaceId": "hud", "components": [ ... ] }
]
"""

logger = logging.getLogger(__name__)

class SmartGlassesPromptStrategy(PromptStrategy):
    """Strategy using the Smart Glasses Optimized System Prompt."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_template_path: Optional[str] = None
    ):
        self._llm_client = llm_client
        self.system_prompt = V2_SMART_GLASSES_PROMPT

    def set_llm_client(self, llm_client: Any) -> None:
        """Set the LLM client."""
        self._llm_client = llm_client

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

    def _sanitize_stream(self, messages: list) -> list:
        """Sanitize all components in an A2UI message stream."""
        for msg in messages:
            if msg.get("type") == "updateComponents":
                if "components" in msg and isinstance(msg["components"], list):
                    msg["components"] = [
                        self._strip_style_properties(c) if isinstance(c, dict) else c
                        for c in msg["components"]
                    ]
        return messages

    @property
    def name(self) -> str:
        """Unique name identifier for this strategy."""
        return "v2_smart_glasses"

    def supports_visual_context(self) -> bool:
        """This strategy supports visual context."""
        return True

    def generate(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
        visual_context: Optional[dict] = None
    ) -> Optional[dict]:
        """Generate A2UI component using the Smart Glasses prompt."""
        
        # Construct User Prompt
        visual_desc = ""
        if visual_context:
            visual_desc = f"\n\n**Visual Context**:\n{json.dumps(visual_context, indent=2)}"

        user_prompt = f"""
**User Intent / Recommendation**:
type: {recommendation.type}
content: "{recommendation.content}"

**Scene Context**:
scene: {scene.name}
time_range: {recommendation.start_time}s - {recommendation.end_time}s
{visual_desc}

**Task**:
Design the AR HUD interface for this recommendation.
"""

        try:
            # Call LLM
            response = self._llm_client.complete(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                temperature=0.7
            )

            # Parse Output using the delimiter
            if "---a2ui_JSON---" in response:
                _, json_part = response.split("---a2ui_JSON---", 1)
                json_part = json_part.strip()
            else:
                # Fallback: try to find the start of JSON list
                start_idx = response.find("[")
                if start_idx != -1:
                    json_part = response[start_idx:]
                else:
                    logger.error("No JSON part found in response")
                    return None

            # Clean markdown code blocks if present
            if json_part.startswith("```json"):
                json_part = json_part[7:]
            if json_part.startswith("```"):
                json_part = json_part[3:]
            if json_part.endswith("```"):
                json_part = json_part[:-3]
            
            json_part = json_part.strip()

            # Parse JSON
            a2ui_messages = json.loads(json_part)

            if not isinstance(a2ui_messages, list):
                 logger.error("Response is not a list of messages")
                 return None

            # Sanitize all components to remove any leaked style properties
            a2ui_messages = self._sanitize_stream(a2ui_messages)

            # For compatibility with legacy pipeline which expects a single component object,
            # we wrap the messages in a special "Component" envelope or return the first relevant component.
            # However, the pipeline seems to expect a single "Component" dict.
            # Let's wrap the whole message stream as a "Surface" component for now, 
            # or return a structure that the Converter can handle.
            
            # Legacy Hack: Create a wrapper component that holds the full A2UI stream
            return {
                "type": "SmartGlassesSurface",
                "id": "generated_root",
                "props": {},
                "metadata": {
                    "a2ui_stream": a2ui_messages,
                    "generated_by": "v2_smart_glasses"
                }
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            logger.debug(f"Failed JSON: {json_part}")
            return None
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None
