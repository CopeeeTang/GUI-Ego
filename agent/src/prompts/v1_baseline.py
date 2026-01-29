"""V1 Baseline prompt strategy - wraps existing two-step approach."""

import logging
from typing import Any, Optional

from .base import PromptStrategy, Recommendation, SceneConfig

logger = logging.getLogger(__name__)


class BaselinePromptStrategy(PromptStrategy):
    """Baseline strategy using the existing two-step approach.

    This strategy wraps the existing ComponentSelector and PropsFiller
    classes to provide backward compatibility with the original pipeline.

    The two-step approach:
    1. Component Selection: Use LLM to select the most appropriate component type
    2. Props Filling: Use LLM to fill in the component properties

    This strategy does NOT support visual context.
    """

    @property
    def name(self) -> str:
        return "v1_baseline"

    @property
    def description(self) -> str:
        return "Two-step approach: Component selection + Props filling (no visual context)"

    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize the baseline strategy.

        Args:
            llm_client: LLM client instance. If None, will be set later via set_llm_client.
        """
        self._llm_client = llm_client
        self._component_selector = None
        self._props_filler = None

    def set_llm_client(self, llm_client: Any) -> None:
        """Set the LLM client (for deferred initialization)."""
        self._llm_client = llm_client
        self._component_selector = None
        self._props_filler = None

    @property
    def component_selector(self):
        """Lazy initialization of ComponentSelector."""
        if self._component_selector is None:
            if self._llm_client is None:
                raise RuntimeError("LLM client not set. Call set_llm_client first.")
            from ..component_selector import ComponentSelector
            self._component_selector = ComponentSelector(self._llm_client)
        return self._component_selector

    @property
    def props_filler(self):
        """Lazy initialization of PropsFiller."""
        if self._props_filler is None:
            if self._llm_client is None:
                raise RuntimeError("LLM client not set. Call set_llm_client first.")
            from ..props_filler import PropsFiller
            self._props_filler = PropsFiller(self._llm_client)
        return self._props_filler

    def supports_visual_context(self) -> bool:
        """Baseline strategy does not support visual context."""
        return False

    def generate(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
        visual_context: Optional[dict] = None,
    ) -> dict:
        """Generate UI component using two-step approach.

        Args:
            recommendation: The AI recommendation to convert to UI.
            scene: Scene configuration with allowed components.
            visual_context: Ignored in baseline strategy.

        Returns:
            Dictionary containing the generated UI component.
        """
        self.validate_inputs(recommendation, scene)

        if visual_context is not None:
            logger.warning(
                "Visual context provided but v1_baseline strategy does not use it. "
                "Consider using v3_with_visual strategy instead."
            )

        # Convert to data_loader types for compatibility
        from ..data_loader import Recommendation as DLRecommendation, SceneConfig as DLSceneConfig

        dl_recommendation = DLRecommendation(
            id=recommendation.id,
            type=recommendation.type,
            content=recommendation.content,
            start_time=recommendation.start_time,
            end_time=recommendation.end_time,
        )

        dl_scene = DLSceneConfig(
            name=scene.name,
            allowed_components=scene.allowed_components,
        )

        # Step 1: Component Selection
        logger.info(f"Step 1: Selecting component for recommendation {recommendation.id}")
        selection = self.component_selector.select_component(dl_recommendation, dl_scene)

        selected_component = selection.get("selected_component")
        if not selected_component:
            raise ValueError(f"Component selection failed: {selection}")

        logger.info(f"Selected component: {selected_component} (confidence: {selection.get('confidence', 'N/A')})")

        # Step 2: Props Filling
        logger.info(f"Step 2: Filling props for {selected_component}")
        component = self.props_filler.generate_component(selected_component, dl_recommendation)

        # Add selection metadata
        component["metadata"] = component.get("metadata", {})
        component["metadata"]["selection"] = {
            "strategy": self.name,
            "selected_component": selected_component,
            "reasoning": selection.get("reasoning"),
            "confidence": selection.get("confidence"),
        }

        return component
