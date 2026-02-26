"""
ExampleLoader - Load preprocessed examples from example/ directory.

This loader replaces DataLoader for pipeline execution, using pre-extracted
frames and structured rawdata.json files instead of raw annotations.

Supports loading user profiles and scene contexts from:
- example/user_profiles.json
- example/scene_contexts.json
"""

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from .schema import Recommendation, SceneConfig

logger = logging.getLogger(__name__)


class ExampleLoader:
    """Load preprocessed examples from example/ directory structure.

    Directory structure:
        example/
            Task2.1/
                P10_Ernesto/
                    sample_001/
                        rawdata.json
                        video/frame_*.jpg
                        signals/*.csv
            Task2.2/
                P10_Ernesto/
                    sample_001/
                        ...
    """

    def __init__(self, base_path: str, participant: str = "P10_Ernesto"):
        """Initialize the example loader.

        Args:
            base_path: Root directory containing example/ folder.
            participant: Participant ID (e.g., "P10_Ernesto").
        """
        self.base_path = Path(base_path)
        self.participant = participant

        # Robustly detect example path by searching up from base_path
        self.example_path = None
        curr = self.base_path.resolve()
        # Search up to 3 levels up for 'example' directory
        for _ in range(4):
            if (curr / "example").exists():
                self.example_path = curr / "example"
                break
            if curr == curr.parent:
                break
            curr = curr.parent

        # Fallback to CWD if still not found
        if not self.example_path and (Path.cwd() / "example").exists():
            self.example_path = Path.cwd() / "example"

        if not self.example_path:
            self.example_path = self.base_path / "example"  # Final fallback

        logger.info(f"ExampleLoader using path: {self.example_path}")

        # Load user profiles and scene contexts
        self.user_profiles: dict = self._load_user_profiles()
        self.scene_contexts: dict = self._load_scene_contexts()

        logger.info(
            f"Loaded {len(self.user_profiles)} user profiles, "
            f"{len(self.scene_contexts)} scene contexts"
        )

        # Scene configurations (inferred from scene_type)
        self.scenes: dict[str, SceneConfig] = {}

        # Loaded samples: (Recommendation, SceneConfig) pairs
        self.samples: list[tuple[Recommendation, SceneConfig]] = []

        # Scan and load all examples
        self._scan_examples()

        logger.info(
            f"ExampleLoader initialized: {len(self.samples)} samples loaded "
            f"across {len(self.scenes)} scene types"
        )

    def _scan_examples(self) -> None:
        """Scan example/ directory and load all rawdata.json files."""
        if not self.example_path.exists():
            logger.warning(f"Example directory not found: {self.example_path}")
            return

        # Scan Task2.1 and Task2.2 directories
        for task_dir in sorted(self.example_path.glob("Task*")):
            # If participant is "all", load all participants; otherwise load specific one
            if self.participant.lower() == "all":
                participant_dirs = sorted(task_dir.glob("P*"))
            else:
                participant_dirs = [task_dir / self.participant]

            for participant_dir in participant_dirs:
                if not participant_dir.exists():
                    logger.debug(f"Participant directory not found: {participant_dir}")
                    continue

                # Load each sample
                for sample_dir in sorted(participant_dir.glob("sample_*")):
                    rawdata_path = sample_dir / "rawdata.json"
                    if not rawdata_path.exists():
                        logger.warning(f"Missing rawdata.json in {sample_dir}")
                        continue

                    try:
                        rawdata = self._load_rawdata(rawdata_path)
                        if rawdata:
                            rec = self._convert_to_recommendation(rawdata, sample_dir)
                            scene_type = rawdata.get("scene_type", "general")

                            # Create scene config if not exists
                            if scene_type not in self.scenes:
                                self.scenes[scene_type] = self._infer_scene_config(scene_type)

                            self.samples.append((rec, self.scenes[scene_type]))
                            logger.debug(f"Loaded sample: {rec.id} ({scene_type})")

                    except Exception as e:
                        logger.error(f"Failed to load {rawdata_path}: {e}")

    def _load_user_profiles(self) -> dict:
        """Load user profiles from example/user_profiles.json.

        Returns:
            Dictionary mapping participant_id to user profile data.
        """
        profiles_path = self.example_path / "user_profiles.json"
        if not profiles_path.exists():
            logger.warning(f"User profiles not found: {profiles_path}")
            return {}

        try:
            with open(profiles_path, "r", encoding="utf-8") as f:
                profiles = json.load(f)
            logger.info(f"Loaded {len(profiles)} user profiles from {profiles_path}")
            return profiles
        except Exception as e:
            logger.error(f"Failed to load user profiles: {e}")
            return {}

    def _load_scene_contexts(self) -> dict:
        """Load scene contexts from example/scene_contexts.json.

        Returns:
            Dictionary mapping participant_id to scene context data.
        """
        contexts_path = self.example_path / "scene_contexts.json"
        if not contexts_path.exists():
            logger.warning(f"Scene contexts not found: {contexts_path}")
            return {}

        try:
            with open(contexts_path, "r", encoding="utf-8") as f:
                contexts = json.load(f)
            logger.info(f"Loaded {len(contexts)} scene contexts from {contexts_path}")
            return contexts
        except Exception as e:
            logger.error(f"Failed to load scene contexts: {e}")
            return {}

    def _get_scene_context_for_time(
        self,
        participant_id: str,
        start_time: float,
        end_time: float,
    ) -> Optional[str]:
        """Get the scene context description for a specific time range.

        Args:
            participant_id: Participant ID (e.g., "P8_MelvynKui").
            start_time: Start time of the sample in seconds.
            end_time: End time of the sample in seconds.

        Returns:
            Scene context description or None if not found.
        """
        context_data = self.scene_contexts.get(participant_id)
        if not context_data:
            return None

        segments = context_data.get("segments", [])
        if not segments:
            # Return summary if no segments defined
            return context_data.get("summary")

        # Find the segment that contains this time range
        mid_time = (start_time + end_time) / 2
        for segment in segments:
            seg_start = segment.get("start_time", 0) or 0
            seg_end = segment.get("end_time")

            # If no end_time, this segment extends to infinity
            if seg_end is None:
                if mid_time >= seg_start:
                    return segment.get("description")
            elif seg_start <= mid_time <= seg_end:
                return segment.get("description")

        # Fallback to summary
        return context_data.get("summary")

    def _load_rawdata(self, path: Path) -> dict | None:
        """Load and validate rawdata.json file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate required fields
            required_fields = ["sample_id", "scene_type", "time_interval", "annotation"]
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field '{field}' in {path}")
                    return None

            return data

        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    def _convert_to_recommendation(
        self,
        rawdata: dict,
        sample_dir: Path
    ) -> Recommendation:
        """Convert rawdata.json to Recommendation object.

        Args:
            rawdata: Parsed rawdata.json content.
            sample_dir: Path to sample directory (for resolving frame paths).

        Returns:
            Recommendation object with metadata including frame paths,
            user profile, and scene context.
        """
        # Extract object labels
        objects = rawdata.get("annotation", {}).get("objects", [])
        object_labels = [obj.get("label", "") for obj in objects if obj.get("label")]

        # Extract task_name from sample_dir path (e.g., Task2.1)
        # sample_dir structure: .../example/Task2.x/Participant/sample_xxx
        task_name = ""
        sample_name = sample_dir.name  # e.g., sample_001
        for part in sample_dir.parts:
            if part.startswith("Task"):
                task_name = part
                break

        # Get participant ID from rawdata or use the loader's participant
        participant_id = rawdata.get("participant", self.participant)

        # Get time interval
        time_interval = rawdata.get("time_interval", {})
        start_time = time_interval.get("start", 0)
        end_time = time_interval.get("end", 0)

        # Load user profile for this participant
        user_profile = self.user_profiles.get(participant_id, {})

        # Load scene context for this time range
        scene_context_desc = self._get_scene_context_for_time(
            participant_id, start_time, end_time
        )

        # Prepare annotation dict for Recommendation.from_annotation()
        annotation_dict = {
            "id": rawdata["sample_id"],
            "type": rawdata.get("scene_type", "general"),
            "text": rawdata.get("annotation", {}).get("text", ""),
            "time_interval": rawdata.get("time_interval", {}),
            "object_list": object_labels,
            "metadata": {
                # Store frame paths (relative to sample_dir)
                "frame_paths": rawdata.get("video", {}).get("frames", []),
                # Store sample directory for absolute path resolution
                "sample_dir": str(sample_dir),
                # Hierarchical path components for output organization
                "task_name": task_name,
                "sample_name": sample_name,
                # Additional metadata
                "expected_response": rawdata.get("annotation", {}).get("expected_response", ""),
                "objects_detail": objects,
                "task": rawdata.get("task", ""),
                "participant": participant_id,
                # User profile and scene context for LLM prompts
                "user_profile": user_profile,
                "scene_context": scene_context_desc,
            }
        }

        # Use Recommendation.from_annotation() for consistent parsing
        return Recommendation.from_annotation(
            annotation_dict,
            rec_id=rawdata["sample_id"]
        )

    def _infer_scene_config(self, scene_type: str) -> SceneConfig:
        """Infer scene configuration from scene_type.

        Args:
            scene_type: Scene type (e.g., "navigation", "shopping", "general").

        Returns:
            SceneConfig with A2UI atomic components.
        """
        # A2UI 原子组件映射 - 与 Preview 渲染器兼容
        scene_components = {
            "navigation": [
                "Card", "Text", "Button", "Icon", "Badge", "Row", "Column"
            ],
            "shopping": [
                "Card", "Text", "Button", "Badge", "List", "Row", "Column", "Icon"
            ],
            "general": [
                "Card", "Text", "Icon", "Badge", "Row", "Column"
            ],
        }

        allowed_components = scene_components.get(
            scene_type,
            scene_components["general"]
        )

        return SceneConfig(
            name=scene_type,
            allowed_components=allowed_components,
            description=f"A2UI atomic components for {scene_type} scene",
        )

    def iter_mvp_data(
        self,
        scenes: list[str] | None = None,
        limit: int = 50,
    ) -> Iterator[tuple[Recommendation, SceneConfig]]:
        """Iterate MVP data samples (interface-compatible with DataLoader).

        Args:
            scenes: List of scene types to filter (None = all scenes).
            limit: Maximum number of samples to yield.

        Yields:
            Tuple of (Recommendation, SceneConfig).
        """
        count = 0
        for rec, scene in self.samples:
            # Filter by scene type if specified
            if scenes is not None and scene.name not in scenes:
                continue

            # Stop at limit
            if count >= limit:
                return

            yield rec, scene
            count += 1

    def get_scene_types(self) -> list[str]:
        """Get list of available scene types."""
        return list(self.scenes.keys())

    def get_sample_count(self, scene_type: str | None = None) -> int:
        """Get total sample count, optionally filtered by scene type."""
        if scene_type is None:
            return len(self.samples)

        return sum(1 for _, scene in self.samples if scene.name == scene_type)
