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

from .example_loader import ExampleLoader
from .llm_client import LLMClient
from .llm import create_client, LLMClientBase
from .component_selector import ComponentSelector
from .props_filler import PropsFiller
from .schema_validator import SchemaValidator
from .schema import Recommendation, SceneConfig

# New modules
from .video.extractor import FrameExtractor
from .video.visual_context import VisualContextGenerator, VisualContextMode, VisualContextCache
from .prompts.base import PromptStrategy
from .prompts.v1_baseline import BaselinePromptStrategy
from .prompts.v2_google_gui import GoogleGUIPromptStrategy
from .prompts.v3_with_visual import VisualPromptStrategy
from .prompts.v2_smart_glasses import SmartGlassesPromptStrategy
from .a2ui.converter import A2UIConverter
from .a2ui.message_builder import A2UIMessageBuilder, A2UISession

logger = logging.getLogger(__name__)

# Strategy registry
STRATEGY_REGISTRY = {
    "v1_baseline": BaselinePromptStrategy,
    "v2_google_gui": GoogleGUIPromptStrategy,
    "v3_with_visual": VisualPromptStrategy,
    "v2_smart_glasses": SmartGlassesPromptStrategy,
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
        visual_mode: str = "direct",
        output_format: str = "legacy",
        video_path: str | None = None,
        num_frames: int = 3,
        google_gui_template: str | None = None,
        # Multi-LLM support
        model_spec: str | None = None,
    ):
        """Initialize the pipeline.

        Args:
            data_path: Path to the dataset (should contain example/ folder).
            output_path: Path for output files.
            participant: Participant ID.
            azure_endpoint: Azure OpenAI endpoint (deprecated, use model_spec).
            azure_api_key: Azure OpenAI API key (deprecated, use model_spec).
            prompt_strategy: Strategy to use ("v1_baseline", "v2_google_gui", "v3_with_visual").
            enable_visual: Whether to enable video frame extraction.
            visual_mode: Visual context mode ("direct" or "description").
            output_format: Output format ("legacy" or "a2ui_standard").
            video_path: Path to video file (auto-detected if None).
            num_frames: Number of frames to extract per recommendation.
            google_gui_template: Path to Google GUI prompt template.
            model_spec: LLM model specification (e.g., "azure:gpt-4o", "gemini:gemini-2.5-pro", "claude:claude-sonnet-4-5").
        """
        self.data_path = Path(data_path)
        self.participant = participant

        # Use ExampleLoader for data loading
        self.data_loader = ExampleLoader(data_path, participant)
        logger.info(f"Using ExampleLoader with {self.data_loader.get_sample_count()} samples")

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.prompt_strategy_name = prompt_strategy
        self.enable_visual = enable_visual
        self.visual_mode = visual_mode
        self.output_format = output_format
        self.num_frames = num_frames
        self.model_spec = model_spec or "azure:gpt-4o"

        # Initialize LLM client using new multi-provider factory
        if model_spec:
            self.llm_client = create_client(model_spec)
            logger.info(f"Using LLM: {model_spec}")
        else:
            # Backward compatibility: use LLMClient wrapper
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
        recommendation: Recommendation,
        scene: SceneConfig,
    ) -> tuple[Recommendation, SceneConfig]:
        """Convert types for prompt strategy compatibility.

        Since we now use unified types from schema module, this is mostly a passthrough.
        """
        return recommendation, scene

    def _extract_visual_context(
        self,
        recommendation: Recommendation,
    ) -> Optional[dict]:
        """Extract visual context for a recommendation.

        Prioritizes pre-extracted frames from metadata, falls back to real-time extraction.
        """
        # Priority 1: Use pre-extracted frames from ExampleLoader
        if hasattr(recommendation, 'metadata') and recommendation.metadata:
            if "frame_paths" in recommendation.metadata and "sample_dir" in recommendation.metadata:
                return self._load_preextracted_frames(recommendation)

        # Priority 2: Real-time extraction (legacy DataLoader or when no pre-extracted frames)
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

    def _load_preextracted_frames(self, recommendation: Recommendation) -> Optional[dict]:
        """Load visual context from pre-extracted frames.

        Args:
            recommendation: Recommendation with metadata containing frame_paths and sample_dir.

        Returns:
            Visual context dict or None if loading fails.
        """
        import base64
        import cv2

        try:
            sample_dir = Path(recommendation.metadata["sample_dir"])
            frame_paths = recommendation.metadata["frame_paths"]

            if not frame_paths:
                logger.warning(f"No pre-extracted frames listed for {recommendation.id}")
                return None

            # Load frames as numpy arrays for VisualContextGenerator
            frames_np = []
            frames_base64 = []

            for frame_rel_path in frame_paths:
                frame_abs_path = sample_dir / frame_rel_path
                if frame_abs_path.exists():
                    # Load image with OpenCV
                    frame = cv2.imread(str(frame_abs_path))
                    if frame is not None:
                        frames_np.append(frame)
                        # Also encode to base64 for direct mode
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        b64_str = base64.b64encode(buffer).decode('utf-8')
                        frames_base64.append(b64_str)
                    else:
                        logger.warning(f"Failed to load frame: {frame_abs_path}")
                else:
                    logger.warning(f"Frame not found: {frame_abs_path}")

            if not frames_np:
                logger.warning(f"No valid pre-extracted frames found for {recommendation.id}")
                return None

            logger.debug(f"Loaded {len(frames_np)} pre-extracted frames for {recommendation.id}")

            # Use VisualContextGenerator if available for consistent processing
            if self.visual_generator:
                context = self.visual_generator.generate_context(frames_np, recommendation)
                return context.to_dict()

            # Fallback: Return frames with base64 encoding for direct mode
            return {
                "mode": "direct",
                "frames_base64": frames_base64,
                "description": f"Loaded {len(frames_base64)} pre-extracted frames",
                "metadata": {
                    "num_frames": len(frames_base64),
                    "encoding": "base64",
                    "format": "jpeg",
                },
            }

        except Exception as e:
            logger.error(f"Failed to load pre-extracted frames for {recommendation.id}: {e}")
            return None

    def _save_component_hierarchical(
        self,
        recommendation: Recommendation,
        component: dict,
    ) -> None:
        """Save a component to the hierarchical output structure.

        Output structure: output/Task2.x/Participant/sample_xxx/{strategy}.json

        Args:
            recommendation: The source recommendation with metadata.
            component: The generated component to save.
        """
        metadata = getattr(recommendation, 'metadata', {}) or {}
        task_name = metadata.get('task_name', '')
        sample_name = metadata.get('sample_name', '')
        participant = metadata.get('participant', self.participant)

        if not task_name or not sample_name:
            # Fallback: try to parse from sample_dir
            sample_dir = metadata.get('sample_dir', '')
            if sample_dir:
                parts = Path(sample_dir).parts
                # Find Task2.x part
                task_idx = next((i for i, p in enumerate(parts) if p.startswith('Task')), -1)
                if task_idx >= 0 and task_idx + 2 < len(parts):
                    task_name = parts[task_idx]
                    participant = parts[task_idx + 1]
                    sample_name = parts[task_idx + 2]

        if not task_name or not sample_name:
            logger.warning(f"Cannot determine hierarchical path for {recommendation.id}, skipping individual save")
            return

        # Build output path: output/Task2.x/Participant/sample_xxx/
        output_sample_dir = self.output_path / task_name / participant / sample_name
        output_sample_dir.mkdir(parents=True, exist_ok=True)

        # Use strategy name + model as filename (e.g., v1_baseline_azure_gpt-4o.json)
        model_suffix = self.model_spec.replace(':', '_')
        component_file = output_sample_dir / f"{self.prompt_strategy_name}_{model_suffix}.json"

        # Add metadata to component
        component["metadata"] = component.get("metadata", {})
        component["metadata"]["strategy"] = self.prompt_strategy_name
        component["metadata"]["model"] = self.model_spec
        component["metadata"]["generated_at"] = datetime.now().isoformat()
        component["metadata"]["visual_mode"] = self.visual_mode if self.enable_visual else None
        component["metadata"]["source_sample"] = f"{task_name}/{participant}/{sample_name}"

        try:
            with open(component_file, "w", encoding="utf-8") as f:
                json.dump(component, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved component to {component_file}")
        except Exception as e:
            logger.error(f"Failed to save component to {component_file}: {e}")

    def process_recommendation(
        self,
        recommendation: Recommendation,
        scene: SceneConfig,
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

                # Save individual component to hierarchical structure
                if save_individual:
                    self._save_component_hierarchical(recommendation, component)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_suffix = f"_{self.prompt_strategy_name}"
        # Add model suffix (e.g., _azure_gpt-4o, _gemini_gemini-3-flash)
        model_suffix = f"_{self.model_spec.replace(':', '_')}"

        # Save summary file
        summary_file = self.output_path / f"a2ui_components{strategy_suffix}{model_suffix}_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Save by scene
        for scene_name, components in scene_results.items():
            if components:
                scene_file = self.output_path / f"a2ui_{scene_name}{strategy_suffix}{model_suffix}_{timestamp}.json"
                with open(scene_file, "w", encoding="utf-8") as f:
                    json.dump(components, f, ensure_ascii=False, indent=2)

        # Save A2UI messages if enabled
        if a2ui_session:
            messages_file = self.output_path / f"a2ui_messages{strategy_suffix}{model_suffix}_{timestamp}.json"
            with open(messages_file, "w", encoding="utf-8") as f:
                json.dump(a2ui_session.export_messages(), f, ensure_ascii=False, indent=2)

        # Print statistics
        print()
        print(f"✅ Pipeline completed!")
        print(f"   Strategy: {self.prompt_strategy_name}")
        print(f"   Model: {self.model_spec}")
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
  # Run with default Azure GPT-4o
  python -m agent.src.pipeline --limit 10

  # Run with Gemini
  python -m agent.src.pipeline --model gemini:gemini-2.5-pro --limit 5

  # Run with Claude
  python -m agent.src.pipeline --model claude:claude-sonnet-4-5 --limit 5

  # Run with Claude thinking model (auto-enables extended thinking)
  python -m agent.src.pipeline --model claude:claude-opus-4-5-thinking --limit 5

  # Run with visual context
  python -m agent.src.pipeline --model gemini:gemini-2.5-flash --strategy v3_with_visual --enable-visual --limit 5

  # Compare strategies
  python -m agent.src.pipeline --compare v1_baseline v3_with_visual --limit 5
        """,
    )

    parser.add_argument(
        "--data-path",
        default="/home/v-tangxin/GUI/agent",
        help="数据集路径 (应包含 example/ 文件夹)",
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
        choices=["v1_baseline", "v2_google_gui", "v3_with_visual", "v2_smart_glasses"],
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
        default="direct",
        help="视觉上下文模式 (默认: direct)",
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
        "--overlay",
        action="store_true",
        help="生成视频叠加预览",
    )
    parser.add_argument(
        "--overlay-duration",
        type=float,
        default=2.0,
        help="UI 显示时长（秒）",
    )
    parser.add_argument(
        "--preview-server-url",
        default="http://localhost:8080",
        help="Preview server URL for overlay rendering",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出",
    )
    parser.add_argument(
        "--model",
        default="azure:gpt-4o",
        help="LLM 模型 (格式: provider:model_name, 例如 azure:gpt-4o, gemini:gemini-2.5-pro, claude:claude-sonnet-4-5)",
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
        model_spec=args.model,
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

    # Run overlay processing if enabled
    if args.overlay:
        print()
        print("🎬 Generating video overlays...")
        from .video.overlay import VideoOverlayProcessor

        overlay_processor = VideoOverlayProcessor(
            preview_server_url=args.preview_server_url,
            overlay_duration=args.overlay_duration,
        )

        # Find generated UI JSON files and their corresponding samples
        output_path = Path(args.output_path)
        data_path = Path(args.data_path) / "example"

        overlay_count = 0
        for task_dir in output_path.iterdir():
            if not task_dir.is_dir() or not task_dir.name.startswith("Task"):
                continue

            for participant_dir in task_dir.iterdir():
                if not participant_dir.is_dir():
                    continue

                for sample_dir in participant_dir.iterdir():
                    if not sample_dir.is_dir() or not sample_dir.name.startswith("sample"):
                        continue

                    # Find UI JSON for this sample
                    ui_json = sample_dir / f"{args.strategy}.json"
                    if not ui_json.exists():
                        continue

                    # Find corresponding source sample directory
                    source_sample = data_path / task_dir.name / participant_dir.name / sample_dir.name
                    if not source_sample.exists():
                        continue

                    result = overlay_processor.process_sample(
                        sample_dir=source_sample,
                        ui_json_path=ui_json,
                        output_dir=output_path,
                        strategy_name=args.strategy,
                    )

                    if result:
                        overlay_count += 1
                        if args.verbose:
                            print(f"   ✓ {result.name}")

        print(f"   Generated {overlay_count} overlay videos")


if __name__ == "__main__":
    main()
