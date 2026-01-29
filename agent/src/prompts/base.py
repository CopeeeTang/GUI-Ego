"""Base class for prompt strategies."""

from abc import ABC, abstractmethod
from typing import Any, Optional

# Import unified schema - single source of truth
from ..schema import Recommendation, SceneConfig


class PromptStrategy(ABC):
    """Abstract base class for prompt strategies.

    This defines the interface that all prompt strategy implementations
    must follow. Each strategy represents a different approach to
    generating UI components from recommendations.

    Subclasses should implement:
    - name: A unique identifier for the strategy
    - generate: The main component generation logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this strategy."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the strategy."""
        return f"Prompt strategy: {self.name}"

    @abstractmethod
    def generate(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
        visual_context: Optional[dict] = None,
    ) -> dict:
        """Generate UI component from recommendation.

        Args:
            recommendation: The AI recommendation to convert to UI.
            scene: Scene configuration with allowed components.
            visual_context: Optional visual context from video frames.

        Returns:
            Dictionary containing the generated UI component in A2UI format.

        Raises:
            ValueError: If generation fails due to invalid input.
            RuntimeError: If LLM call fails.
        """
        pass

    def validate_inputs(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
    ) -> None:
        """Validate inputs before generation.

        Args:
            recommendation: The recommendation to validate.
            scene: The scene configuration to validate.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not recommendation.content:
            raise ValueError("Recommendation content cannot be empty")

        if not scene.allowed_components:
            raise ValueError("Scene must have at least one allowed component")

    def supports_visual_context(self) -> bool:
        """Check if this strategy supports visual context.

        Returns:
            True if the strategy can use visual context, False otherwise.
        """
        return False


class PromptStrategyRegistry:
    """Registry for prompt strategies.

    This registry allows dynamic registration and lookup of
    prompt strategies by name.
    """

    _strategies: dict[str, type[PromptStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: type[PromptStrategy]) -> type[PromptStrategy]:
        """Register a prompt strategy class.

        Can be used as a decorator:
            @PromptStrategyRegistry.register
            class MyStrategy(PromptStrategy):
                ...

        Args:
            strategy_class: The strategy class to register.

        Returns:
            The registered class (for decorator use).
        """
        # Create an instance to get the name
        instance = strategy_class.__new__(strategy_class)
        if hasattr(strategy_class, 'name') and isinstance(strategy_class.name, property):
            # Handle property-based name
            name = strategy_class.name.fget(instance)
        else:
            # Try to instantiate to get name
            try:
                temp_instance = strategy_class()
                name = temp_instance.name
            except TypeError:
                # Can't instantiate without args, use class name
                name = strategy_class.__name__.lower().replace("promptstrategy", "")

        cls._strategies[name] = strategy_class
        return strategy_class

    @classmethod
    def get(cls, name: str) -> type[PromptStrategy]:
        """Get a strategy class by name.

        Args:
            name: The strategy name.

        Returns:
            The strategy class.

        Raises:
            KeyError: If strategy is not found.
        """
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise KeyError(
                f"Strategy '{name}' not found. Available: {available}"
            )
        return cls._strategies[name]

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> PromptStrategy:
        """Create a strategy instance by name.

        Args:
            name: The strategy name.
            **kwargs: Arguments to pass to the strategy constructor.

        Returns:
            An instance of the strategy.
        """
        strategy_class = cls.get(name)
        return strategy_class(**kwargs)
