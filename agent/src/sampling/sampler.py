"""Main sampler module for generating Ego-Dataset examples."""

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .data_discovery import DataDiscovery, AnnotationEntry
from .signal_extractor import SignalExtractor
from .video_clipper import VideoClipper

logger = logging.getLogger(__name__)


class EgoDatasetSampler:
    """Sample and generate structured examples from Ego-Dataset."""

    def __init__(
        self,
        data_root: str | Path,
        output_root: str | Path = "example",
        signal_window: float = 5.0,
        num_keyframes: int = 3,
    ):
        """Initialize the sampler.

        Args:
            data_root: Root directory of the ego-dataset (e.g., data/ego-dataset/data/)
            output_root: Root directory for output examples
            signal_window: Time window (seconds) to add before/after annotations for signals
            num_keyframes: Number of keyframes to extract per sample
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.signal_window = signal_window
        self.num_keyframes = num_keyframes

        self.discovery = DataDiscovery(self.data_root)

        # Cache for video clippers and signal extractors
        self._video_clippers: dict[str, VideoClipper] = {}
        self._signal_extractors: dict[str, SignalExtractor] = {}

    def _get_video_clipper(self, participant: str) -> Optional[VideoClipper]:
        """Get or create a VideoClipper for a participant."""
        if participant in self._video_clippers:
            return self._video_clippers[participant]

        video_path = self.discovery.get_participant_video_path(participant)
        if video_path is None:
            logger.warning(f"No video found for {participant}")
            return None

        try:
            clipper = VideoClipper(video_path)
            self._video_clippers[participant] = clipper
            return clipper
        except Exception as e:
            logger.error(f"Error creating VideoClipper for {participant}: {e}")
            return None

    def _get_signal_extractor(self, participant: str) -> Optional[SignalExtractor]:
        """Get or create a SignalExtractor for a participant."""
        if participant in self._signal_extractors:
            return self._signal_extractors[participant]

        signals_config = self.discovery.get_participant_signals_dir(participant)

        try:
            extractor = SignalExtractor(signals_config)
            self._signal_extractors[participant] = extractor
            return extractor
        except Exception as e:
            logger.error(f"Error creating SignalExtractor for {participant}: {e}")
            return None

    def _generate_sample_id(self, task: str, participant: str, index: int) -> str:
        """Generate a unique sample ID."""
        # Extract participant number
        p_num = participant.split("_")[0].lower()
        return f"task{task}_{p_num}_{index:03d}"

    def _build_rawdata_json(
        self,
        entry: AnnotationEntry,
        sample_id: str,
        signals_saved: dict,
        video_result: dict,
    ) -> dict:
        """Build the rawdata.json structure.

        Args:
            entry: Annotation entry
            sample_id: Unique sample identifier
            signals_saved: Dictionary with paths to saved signal files
            video_result: Dictionary with clip and frame paths

        Returns:
            Dictionary representing rawdata.json
        """
        # Determine scene type based on annotation content
        scene_type = self._infer_scene_type(entry)

        # Build signals info
        signals_info = {
            "gaze": {
                "window": f"±{self.signal_window}s",
                "path": "signals/gaze.csv" if signals_saved.get("gaze") else None,
                "available": signals_saved.get("gaze") is not None,
            },
            "heart_rate": {
                "window": f"±{self.signal_window}s",
                "path": "signals/hr.csv" if signals_saved.get("hr") else None,
                "available": signals_saved.get("hr") is not None,
            },
            "eda": {
                "window": f"±{self.signal_window}s",
                "path": "signals/eda.csv" if signals_saved.get("eda") else None,
                "available": signals_saved.get("eda") is not None,
            },
        }

        # Build video info
        frame_paths = [
            f"video/{p.name}" for p in video_result.get("frame_paths", [])
        ]
        video_info = {
            "clip_path": "video/clip.mp4" if video_result.get("clip_path") else None,
            "frames": frame_paths,
            "available": video_result.get("clip_path") is not None,
        }

        # Build annotation info
        annotation_info = {
            "text": entry.text,
            "expected_response": entry.expected_response,
            "objects": entry.objects,
        }
        if entry.annotation_type:
            annotation_info["type"] = entry.annotation_type

        rawdata = {
            "sample_id": sample_id,
            "task": entry.task,
            "participant": entry.participant,
            "scene_type": scene_type,
            "time_interval": entry.time_interval,
            "duration": entry.duration,
            "annotation": annotation_info,
            "user_profile": {
                "participant_id": entry.participant,
            },
            "scene_context": {
                "location": "清华大学校园",
                "activity": scene_type,
            },
            "signals": signals_info,
            "video": video_info,
        }

        return rawdata

    def _infer_scene_type(self, entry: AnnotationEntry) -> str:
        """Infer scene type from annotation content."""
        text = entry.text.lower()

        # Navigation related
        if any(kw in text for kw in ["导航", "路线", "怎么走", "去哪", "位置", "地图"]):
            return "navigation"

        # Food/shopping related
        if any(kw in text for kw in ["食物", "吃", "卡路里", "超市", "购物", "买", "酸奶", "三明治"]):
            return "shopping"

        # Information query
        if any(kw in text for kw in ["是什么", "告诉我", "帮我看", "识别", "翻译"]):
            return "information_query"

        # Task assistance
        if any(kw in text for kw in ["提醒", "添加", "设置", "拍照", "保存"]):
            return "task_assistance"

        # Use annotation type for Task 2.2
        if entry.annotation_type:
            return entry.annotation_type

        return "general"

    def generate_sample(
        self,
        entry: AnnotationEntry,
        sample_index: int,
    ) -> Optional[Path]:
        """Generate a single sample from an annotation entry.

        Args:
            entry: Annotation entry to generate sample from
            sample_index: Index for sample ID generation

        Returns:
            Path to the sample directory, or None if failed
        """
        sample_id = self._generate_sample_id(entry.task, entry.participant, sample_index)

        # Create output directory structure
        task_dir = f"Task{entry.task}"
        sample_dir = self.output_root / task_dir / entry.participant / f"sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating sample {sample_id} at {sample_dir}")

        # Extract video clip and keyframes
        video_result = {"clip_path": None, "frame_paths": []}
        clipper = self._get_video_clipper(entry.participant)
        if clipper:
            try:
                video_result = clipper.generate_clip_and_frames(
                    entry.start_time,
                    entry.end_time,
                    sample_dir,
                    num_frames=self.num_keyframes,
                )
            except Exception as e:
                logger.error(f"Error generating video for {sample_id}: {e}")

        # Extract signals
        signals_saved = {"gaze": None, "hr": None, "eda": None}
        extractor = self._get_signal_extractor(entry.participant)
        if extractor:
            try:
                signals_dir = sample_dir / "signals"
                signals_saved = extractor.save_signals_to_csv(
                    entry.start_time,
                    entry.end_time,
                    signals_dir,
                    window=self.signal_window,
                )
            except Exception as e:
                logger.error(f"Error extracting signals for {sample_id}: {e}")

        # Build and save rawdata.json
        rawdata = self._build_rawdata_json(entry, sample_id, signals_saved, video_result)

        rawdata_path = sample_dir / "rawdata.json"
        with open(rawdata_path, "w", encoding="utf-8") as f:
            json.dump(rawdata, f, ensure_ascii=False, indent=2)

        logger.info(f"Generated sample at {sample_dir}")
        return sample_dir

    def generate_demo(self, task: str = "2.1") -> Optional[Path]:
        """Generate a single demo sample for confirmation.

        Args:
            task: Task identifier ("2.1" or "2.2")

        Returns:
            Path to the demo sample directory
        """
        logger.info(f"Generating demo sample for Task {task}")

        # Load annotations
        all_annotations = self.discovery.load_all_annotations(task)

        if not all_annotations:
            logger.error(f"No annotations found for Task {task}")
            return None

        # Pick the first participant with annotations
        participant = list(all_annotations.keys())[0]
        entries = all_annotations[participant]

        if not entries:
            logger.error(f"No entries found for {participant}")
            return None

        # Generate the first sample
        return self.generate_sample(entries[0], sample_index=1)

    def batch_generate(
        self,
        task: str,
        count: int = 100,
        random_seed: int = 42,
    ) -> list[Path]:
        """Generate batch samples for a task.

        Args:
            task: Task identifier ("2.1" or "2.2")
            count: Number of samples to generate
            random_seed: Random seed for reproducibility

        Returns:
            List of paths to generated sample directories
        """
        random.seed(random_seed)
        logger.info(f"Generating {count} samples for Task {task}")

        # Load all annotations
        all_annotations = self.discovery.load_all_annotations(task)

        if not all_annotations:
            logger.error(f"No annotations found for Task {task}")
            return []

        # Calculate quota per participant
        participants = list(all_annotations.keys())
        total_available = sum(len(entries) for entries in all_annotations.values())

        if total_available < count:
            logger.warning(
                f"Only {total_available} samples available, reducing count from {count}"
            )
            count = total_available

        # Distribute samples across participants proportionally
        samples_per_participant = {}
        remaining = count

        for participant in participants:
            available = len(all_annotations[participant])
            # Proportional allocation
            quota = max(1, int(count * available / total_available))
            quota = min(quota, available, remaining)
            samples_per_participant[participant] = quota
            remaining -= quota

        # Distribute any remaining samples
        while remaining > 0:
            for participant in participants:
                if remaining <= 0:
                    break
                available = len(all_annotations[participant])
                current = samples_per_participant[participant]
                if current < available:
                    samples_per_participant[participant] += 1
                    remaining -= 1

        logger.info(f"Sample distribution: {samples_per_participant}")

        # Generate samples
        generated_paths = []
        global_index = 1

        # Flatten selected entries for progress bar
        all_selected = []
        for participant, quota in samples_per_participant.items():
            entries = all_annotations[participant]
            if len(entries) > quota:
                all_selected.extend(random.sample(entries, quota))
            else:
                all_selected.extend(entries)

        # Generate samples with progress bar
        with tqdm(total=len(all_selected), desc=f"Generating Task {task}") as pbar:
            for entry in all_selected:
                try:
                    path = self.generate_sample(entry, global_index)
                    if path:
                        generated_paths.append(path)
                        global_index += 1
                except Exception as e:
                    logger.error(f"Error generating sample: {e}")
                finally:
                    pbar.update(1)

        logger.info(f"Generated {len(generated_paths)} samples for Task {task}")
        return generated_paths

    def get_statistics(self) -> dict:
        """Get statistics about available data.

        Returns:
            Dictionary with dataset statistics
        """
        return self.discovery.get_statistics()
