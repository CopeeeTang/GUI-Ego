"""
Generative UI Pipeline

端到端 Pipeline：从标注数据生成 A2UI JSON
支持多种 Prompt 策略和视觉上下文增强
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from tqdm import tqdm

from .data_loader import DataLoader, Recommendation as DLRecommendation, SceneConfig as DLSceneConfig
from .llm_client import LLMClient
from .component_selector import ComponentSelector
from .props_filler import PropsFiller
from .schema_validator import SchemaValidator

# New modules
from .video.extractor import FrameExtractor
from .video.visual_context import VisualContextGenerator, VisualContextMode, VisualContextCache
from .prompts.base import PromptStrategy, Recommendation, SceneConfig
from .prompts.v1_baseline import BaselinePromptStrategy
from .prompts.v2_google_gui import GoogleGUIPromptStrategy
from .prompts.v3_with_visual import VisualPromptStrategy
from .a2ui.converter import A2UIConverter
from .a2ui.message_builder import A2UIMessageBuilder, A2UISession

logger = logging.getLogger(__name__)

# Strategy registry
STRATEGY_REGISTRY = {
    "v1_baseline": BaselinePromptStrategy,
    "v2_google_gui": GoogleGUIPromptStrategy,
    "v3_with_visual": VisualPromptStrategy,
}


class GenerativeUIPipeline:
    """Generative UI Pipeline with video and multi-strategy support"""

    def __init__(
        self,
        data_path: str,
        output_path: str,
        participant: str = "P1_YuePan",
        azure_endpoint: str | None = None,
        azure_api_key: str | None = None,
        # New configuration options
        prompt_strategy: str = "v1_baseline",
        enable_visual: bool = False,
        visual_mode: str = "description",
        output_format: str = "legacy",
        video_path: str | None = None,
        num_frames: int = 3,
        google_gui_template: str | None = None,
    ):
        """Initialize the pipeline.

        Args:
            data_path: Path to the dataset.
            output_path: Path for output files.
            participant: Participant ID.
            azure_endpoint: Azure OpenAI endpoint.
            azure_api_key: Azure OpenAI API key.
            prompt_strategy: Strategy to use ("v1_baseline", "v2_google_gui", "v3_with_visual").
            enable_visual: Whether to enable video frame extraction.
            visual_mode: Visual context mode ("direct" or "description").
            output_format: Output format ("legacy" or "a2ui_standard").
            video_path: Path to video file (auto-detected if None).
            num_frames: Number of frames to extract per recommendation.
            google_gui_template: Path to Google GUI prompt template.
        """
        self.data_path = Path(data_path)
        self.participant = participant
        self.data_loader = DataLoader(data_path, participant)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.prompt_strategy_name = prompt_strategy
        self.enable_visual = enable_visual
        self.visual_mode = visual_mode
        self.output_format = output_format
        self.num_frames = num_frames

        # Initialize LLM client
        self.llm_client = LLMClient(
            endpoint=azure_endpoint,
            api_key=azure_api_key,
        )

        # Initialize components (for backward compatibility)
        self.component_selector = ComponentSelector(self.llm_client)
        self.props_filler = PropsFiller(self.llm_client)
        self.validator = SchemaValidator()

        # Initialize prompt strategy
        self.strategy = self._init_strategy(prompt_strategy, google_gui_template)

        # Initialize video processing (if enabled)
        self.frame_extractor: Optional[FrameExtractor] = None
        self.visual_generator: Optional[VisualContextGenerator] = None
        self.visual_cache = VisualContextCache(max_size=100)

        if enable_visual:
            self._init_video_processing(video_path)

        # Initialize A2UI converter (if using standard format)
        self.a2ui_converter: Optional[A2UIConverter] = None
        self.a2ui_builder: Optional[A2UIMessageBuilder] = None

        if output_format == "a2ui_standard":
            self.a2ui_converter = A2UIConverter()
            self.a2ui_builder = A2UIMessageBuilder()

        logger.info(
            f"Pipeline initialized: strategy={prompt_strategy}, "
            f"visual={enable_visual}, format={output_format}"
        )

    def _init_strategy(
        self,
        strategy_name: str,
        google_gui_template: str | None = None,
    ) -> PromptStrategy:
        """Initialize the prompt strategy."""
        if strategy_name not in STRATEGY_REGISTRY:
            available = ", ".join(STRATEGY_REGISTRY.keys())
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

        strategy_class = STRATEGY_REGISTRY[strategy_name]

        if strategy_name == "v2_google_gui":
            strategy = strategy_class(
                llm_client=self.llm_client,
                prompt_template_path=google_gui_template,
            )
        else:
            strategy = strategy_class(llm_client=self.llm_client)

        # Set LLM client if needed
        if hasattr(strategy, 'set_llm_client'):
            strategy.set_llm_client(self.llm_client)

        return strategy

    def _init_video_processing(self, video_path: str | None = None):
        """Initialize video processing components."""
        # Find video file
        if video_path:
            video_file = Path(video_path)
        else:
            video_file = FrameExtractor.find_video_for_participant(
                self.data_path / "data",
                self.participant,
            )

        if video_file and video_file.exists():
            try:
                self.frame_extractor = FrameExtractor(video_file)
                self.visual_generator = VisualContextGenerator(
                    llm_client=self.llm_client,
                    mode=self.visual_mode,
                )
                logger.info(f"Video processing initialized: {video_file}")
            except Exception as e:
                logger.error(f"Failed to initialize video processing: {e}")
                self.frame_extractor = None
                self.visual_generator = None
        else:
            logger.warning(f"Video file not found for participant {self.participant}")

    def _convert_to_prompt_types(
        self,
        recommendation: DLRecommendation,
        scene: DLSceneConfig,
    ) -> tuple[Recommendation, SceneConfig]:
        """Convert data_loader types to prompts types."""
        rec = Recommendation(
            id=recommendation.id,
            type=recommendation.type,
            content=recommendation.content,
            start_time=recommendation.start_time,
            end_time=recommendation.end_time,
        )

        sc = SceneConfig(
            name=scene.name,
            allowed_components=scene.allowed_components,
        )

        return rec, sc

    def _extract_visual_context(
        self,
        recommendation: Recommendation,
    ) -> Optional[dict]:
        """Extract visual context for a recommendation."""
        if not self.frame_extractor or not self.visual_generator:
            return None

        # Check cache first
        cached = self.visual_cache.get(
            str(self.frame_extractor.video_path),
            recommendation.start_time,
            recommendation.end_time,
            self.num_frames,
            self.visual_mode,
        )
        if cached:
            return cached.to_dict()

        try:
            # Extract frames
            frames = self.frame_extractor.extract_frames(
                recommendation.start_time,
                recommendation.end_time,
                num_frames=self.num_frames,
            )

            if not frames:
                logger.warning(f"No frames extracted for recommendation {recommendation.id}")
                return None

            # Generate visual context
            context = self.visual_generator.generate_context(frames, recommendation)

            # Cache the result
            self.visual_cache.set(
                str(self.frame_extractor.video_path),
                recommendation.start_time,
                recommendation.end_time,
                self.num_frames,
                self.visual_mode,
                context,
            )

            return context.to_dict()

        except Exception as e:
            logger.error(f"Failed to extract visual context: {e}")
            return None

    def process_recommendation(
        self,
        recommendation: DLRecommendation,
        scene: DLSceneConfig,
    ) -> dict[str, Any] | None:
        """Process a single recommendation and generate A2UI component."""
        try:
            # Convert types
            rec, sc = self._convert_to_prompt_types(recommendation, scene)

            # Extract visual context if enabled
            visual_context = None
            if self.enable_visual:
                visual_context = self._extract_visual_context(rec)

            # Generate component using strategy
            component = self.strategy.generate(rec, sc, visual_context)

            if not component:
                logger.warning(f"Strategy returned no component for: {recommendation.content[:30]}...")
                return None

            # Convert to A2UI standard format if configured
            if self.a2ui_converter and self.output_format == "a2ui_standard":
                component = self.a2ui_converter.convert(component)

            # Validate
            is_valid, error = self.validator.validate_component(component)
            if not is_valid:
                logger.warning(f"Component validation failed: {error}")
                # Still return the component but mark as unvalidated
                component["metadata"] = component.get("metadata", {})
                component["metadata"]["validation_error"] = error

            return component

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return None

    def run(
        self,
        scenes: list[str] | None = None,
        limit: int = 50,
        save_individual: bool = True,
        save_a2ui_messages: bool = False,
    ) -> list[dict[str, Any]]:
        """Run the pipeline.

        Args:
            scenes: List of scenes to process.
            limit: Maximum number of recommendations to process.
            save_individual: Whether to save individual component files.
            save_a2ui_messages: Whether to save A2UI message sequences.

        Returns:
            List of generated components.
        """
        if scenes is None:
            scenes = ["navigation", "shopping"]

        print(f"🚀 Starting Generative UI Pipeline")
        print(f"   Strategy: {self.prompt_strategy_name}")
        print(f"   Visual Context: {'Enabled (' + self.visual_mode + ')' if self.enable_visual else 'Disabled'}")
        print(f"   Output Format: {self.output_format}")
        print(f"   Scenes: {scenes}")
        print(f"   Limit: {limit}")
        print()

        results = []
        scene_results = {scene: [] for scene in scenes}
        a2ui_session = A2UISession(self.a2ui_builder) if save_a2ui_messages else None

        # Iterate and process
        data_iter = list(self.data_loader.iter_mvp_data(scenes, limit))

        for recommendation, scene in tqdm(data_iter, desc="Processing"):
            component = self.process_recommendation(recommendation, scene)
            if component:
                results.append(component)
                scene_results[scene.name].append(component)

                # Add to A2UI session if enabled
                if a2ui_session:
                    a2ui_session.create_surface([component])

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_suffix = f"_{self.prompt_strategy_name}"

        # Save summary file
        summary_file = self.output_path / f"a2ui_components{strategy_suffix}_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Save by scene
        for scene_name, components in scene_results.items():
            if components:
                scene_file = self.output_path / f"a2ui_{scene_name}{strategy_suffix}_{timestamp}.json"
                with open(scene_file, "w", encoding="utf-8") as f:
                    json.dump(components, f, ensure_ascii=False, indent=2)

        # Save A2UI messages if enabled
        if a2ui_session:
            messages_file = self.output_path / f"a2ui_messages{strategy_suffix}_{timestamp}.json"
            with open(messages_file, "w", encoding="utf-8") as f:
                json.dump(a2ui_session.export_messages(), f, ensure_ascii=False, indent=2)

        # Print statistics
        print()
        print(f"✅ Pipeline completed!")
        print(f"   Strategy: {self.prompt_strategy_name}")
        print(f"   Total components: {len(results)}")
        for scene_name, components in scene_results.items():
            print(f"   - {scene_name}: {len(components)}")
        print(f"   Output: {summary_file}")

        if self.enable_visual and self.frame_extractor:
            print(f"   Video: {self.frame_extractor.video_path.name}")
            print(f"   Visual mode: {self.visual_mode}")

        return results

    def compare_strategies(
        self,
        strategies: list[str],
        scenes: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, list[dict]]:
        """Compare multiple strategies on the same data.

        Args:
            strategies: List of strategy names to compare.
            scenes: Scenes to process.
            limit: Number of recommendations to process.

        Returns:
            Dictionary mapping strategy names to their results.
        """
        if scenes is None:
            scenes = ["navigation", "shopping"]

        print(f"🔬 Comparing strategies: {strategies}")
        print()

        comparison_results = {}

        for strategy_name in strategies:
            print(f"\n--- Running {strategy_name} ---")

            # Reinitialize with new strategy
            self.prompt_strategy_name = strategy_name
            self.strategy = self._init_strategy(
                strategy_name,
                google_gui_template=None,  # Would need to pass this properly
            )

            # Enable visual for v3
            if strategy_name == "v3_with_visual" and not self.frame_extractor:
                self._init_video_processing(None)

            results = self.run(scenes=scenes, limit=limit)
            comparison_results[strategy_name] = results

        # Save comparison summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = self.output_path / f"strategy_comparison_{timestamp}.json"

        summary = {
            "strategies": strategies,
            "scenes": scenes,
            "limit": limit,
            "results": {
                name: {
                    "count": len(results),
                    "components": results,
                }
                for name, results in comparison_results.items()
            },
        }

        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n📊 Comparison saved to: {comparison_file}")

        return comparison_results


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Glasses Generative UI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default v1_baseline strategy
  python -m agent.src.pipeline --limit 10

  # Run with v3_with_visual strategy and video
  python -m agent.src.pipeline --strategy v3_with_visual --enable-visual --limit 5

  # Compare strategies
  python -m agent.src.pipeline --compare v1_baseline v3_with_visual --limit 5

  # Output in A2UI standard format
  python -m agent.src.pipeline --strategy v3_with_visual --output-format a2ui_standard
        """,
    )

    parser.add_argument(
        "--data-path",
        default="/home/v-tangxin/GUI/data/ego-dataset",
        help="数据集路径",
    )
    parser.add_argument(
        "--output-path",
        default="/home/v-tangxin/GUI/agent/output",
        help="输出路径",
    )
    parser.add_argument(
        "--participant",
        default="P1_YuePan",
        help="被试 ID",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["navigation", "shopping"],
        help="要处理的场景",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="最大处理数量",
    )

    # New arguments
    parser.add_argument(
        "--strategy",
        choices=["v1_baseline", "v2_google_gui", "v3_with_visual"],
        default="v1_baseline",
        help="Prompt 策略",
    )
    parser.add_argument(
        "--enable-visual",
        action="store_true",
        help="启用视频帧提取和视觉上下文",
    )
    parser.add_argument(
        "--visual-mode",
        choices=["direct", "description"],
        default="description",
        help="视觉上下文模式",
    )
    parser.add_argument(
        "--output-format",
        choices=["legacy", "a2ui_standard"],
        default="legacy",
        help="输出格式",
    )
    parser.add_argument(
        "--video-path",
        help="视频文件路径（自动检测如果不指定）",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=3,
        help="每个推荐提取的帧数",
    )
    parser.add_argument(
        "--google-gui-template",
        help="Google GUI Prompt 模板路径",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="比较多个策略",
    )
    parser.add_argument(
        "--save-a2ui-messages",
        action="store_true",
        help="保存 A2UI 消息序列",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create pipeline
    pipeline = GenerativeUIPipeline(
        data_path=args.data_path,
        output_path=args.output_path,
        participant=args.participant,
        prompt_strategy=args.strategy,
        enable_visual=args.enable_visual,
        visual_mode=args.visual_mode,
        output_format=args.output_format,
        video_path=args.video_path,
        num_frames=args.num_frames,
        google_gui_template=args.google_gui_template,
    )

    # Run comparison or single strategy
    if args.compare:
        pipeline.compare_strategies(
            strategies=args.compare,
            scenes=args.scenes,
            limit=args.limit,
        )
    else:
        pipeline.run(
            scenes=args.scenes,
            limit=args.limit,
            save_a2ui_messages=args.save_a2ui_messages,
        )


if __name__ == "__main__":
    main()
