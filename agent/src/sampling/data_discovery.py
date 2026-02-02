"""Data discovery module for scanning and loading Ego-Dataset annotations."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AnnotationEntry:
    """Unified annotation entry for both Task 2.1 and Task 2.2."""

    task: str
    participant: str
    start_time: float
    end_time: float
    text: str
    expected_response: str = ""
    annotation_type: str = ""  # For Task 2.2: recommendation type
    objects: list = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def time_interval(self) -> dict:
        return {"start": self.start_time, "end": self.end_time}


class DataDiscovery:
    """Discover and load annotation data from Ego-Dataset."""

    def __init__(self, data_root: str | Path):
        """Initialize data discovery.

        Args:
            data_root: Root directory of the ego-dataset (e.g., data/ego-dataset/data/)
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

    def discover_participants(self) -> list[str]:
        """Discover all participant directories.

        Returns:
            List of participant directory names (e.g., ["P1_YuePan", "P2_EduardusTjitrahardja"])
        """
        participants = []
        for item in sorted(self.data_root.iterdir()):
            if item.is_dir() and item.name.startswith("P"):
                participants.append(item.name)
        logger.info(f"Discovered {len(participants)} participants")
        return participants

    def find_annotation_files(self, task: str) -> dict[str, list[Path]]:
        """Find annotation files for a specific task.

        Args:
            task: Task identifier ("2.1" or "2.2")

        Returns:
            Dictionary mapping participant names to their annotation file paths
        """
        annotations = {}
        pattern = f"*2.1*.json" if task == "2.1" else f"*2.2*.json"

        for participant in self.discover_participants():
            annotation_dir = self.data_root / participant / "annotation"
            if annotation_dir.exists():
                files = list(annotation_dir.glob(pattern))
                if files:
                    annotations[participant] = files

        logger.info(f"Found Task {task} annotations for {len(annotations)} participants")
        return annotations

    def load_task_2_1_annotations(self, participant: str) -> list[AnnotationEntry]:
        """Load Task 2.1 annotations for a participant.

        Task 2.1 format:
        - start/end: timestamps in seconds
        - text: user query/intent
        - expected response: expected AI response
        - objects: list of object bounding boxes

        Args:
            participant: Participant directory name

        Returns:
            List of AnnotationEntry objects
        """
        annotation_dir = self.data_root / participant / "annotation"
        files = list(annotation_dir.glob("*2.1*.json"))

        entries = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item in data:
                    entry = AnnotationEntry(
                        task="2.1",
                        participant=participant,
                        start_time=float(item.get("start", 0)),
                        end_time=float(item.get("end", 0)),
                        text=item.get("text", ""),
                        expected_response=item.get("expected response", ""),
                        objects=item.get("objects", []),
                        raw_data=item,
                    )
                    if entry.duration > 0:
                        entries.append(entry)

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.debug(f"Loaded {len(entries)} Task 2.1 entries for {participant}")
        return entries

    def load_task_2_2_annotations(self, participant: str) -> list[AnnotationEntry]:
        """Load Task 2.2 annotations for a participant.

        Task 2.2 format:
        - start_time/end_time: segment timestamps
        - accepted_recommendation_list[].time_interval: [start, end] array
        - accepted_recommendation_list[].type: recommendation type
        - accepted_recommendation_list[].original_content: recommendation text

        Args:
            participant: Participant directory name

        Returns:
            List of AnnotationEntry objects (one per accepted recommendation)
        """
        annotation_dir = self.data_root / participant / "annotation"
        files = list(annotation_dir.glob("*2.2*.json"))

        entries = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for segment in data:
                    accepted_list = segment.get("accepted_recommendation_list", [])
                    for rec in accepted_list:
                        time_interval = rec.get("time_interval", [])
                        if len(time_interval) >= 2:
                            start_time = float(time_interval[0])
                            end_time = float(time_interval[1])
                        else:
                            # Fallback to segment times
                            start_time = float(segment.get("start_time", 0))
                            end_time = float(segment.get("end_time", 0))

                        entry = AnnotationEntry(
                            task="2.2",
                            participant=participant,
                            start_time=start_time,
                            end_time=end_time,
                            text=rec.get("original_content", ""),
                            expected_response=rec.get("modified_content", ""),
                            annotation_type=rec.get("type", ""),
                            objects=self._extract_objects_from_rec(rec),
                            raw_data=rec,
                        )
                        if entry.duration > 0:
                            entries.append(entry)

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.debug(f"Loaded {len(entries)} Task 2.2 entries for {participant}")
        return entries

    def _extract_objects_from_rec(self, rec: dict) -> list:
        """Extract object annotations from Task 2.2 recommendation.

        Args:
            rec: Recommendation dictionary

        Returns:
            List of object annotations
        """
        objects = []
        object_list = rec.get("object_list", {})

        if isinstance(object_list, dict):
            for timestamp_key, obj_list in object_list.items():
                if isinstance(obj_list, list):
                    for obj in obj_list:
                        objects.append({
                            "timestamp_key": timestamp_key,
                            "x": obj.get("x"),
                            "y": obj.get("y"),
                            "width": obj.get("width"),
                            "height": obj.get("height"),
                            "label": obj.get("instance", obj.get("label", "")),
                        })
        elif isinstance(object_list, list):
            objects = object_list

        return objects

    def load_all_annotations(self, task: str) -> dict[str, list[AnnotationEntry]]:
        """Load all annotations for a task across all participants.

        Args:
            task: Task identifier ("2.1" or "2.2")

        Returns:
            Dictionary mapping participant names to their annotation entries
        """
        all_annotations = {}
        loader = self.load_task_2_1_annotations if task == "2.1" else self.load_task_2_2_annotations

        for participant in self.discover_participants():
            entries = loader(participant)
            if entries:
                all_annotations[participant] = entries

        total = sum(len(e) for e in all_annotations.values())
        logger.info(f"Loaded {total} total entries for Task {task} from {len(all_annotations)} participants")
        return all_annotations

    def get_participant_video_path(self, participant: str) -> Optional[Path]:
        """Find the main video file for a participant.

        Args:
            participant: Participant directory name

        Returns:
            Path to the video file, or None if not found
        """
        participant_dir = self.data_root / participant

        # Priority 1: Merged video in "Timeseries Data + Scene Video/"
        timeseries_dir = participant_dir / "Timeseries Data + Scene Video"
        if timeseries_dir.exists():
            for session_dir in timeseries_dir.iterdir():
                if session_dir.is_dir():
                    mp4_files = list(session_dir.glob("*.mp4"))
                    if mp4_files:
                        return mp4_files[0]

        # Priority 2: Concatenated video in raw_data
        raw_data_dir = participant_dir / "raw_data"
        if raw_data_dir.exists():
            for session_dir in raw_data_dir.iterdir():
                if session_dir.is_dir():
                    # Look for concatenated/compressed video first
                    concat_videos = list(session_dir.glob("*concat*.mp4")) + \
                                    list(session_dir.glob("*compress*.mp4"))
                    if concat_videos:
                        return concat_videos[0]

                    # Fallback to scene camera videos
                    scene_videos = list(session_dir.glob("Neon Scene Camera*.mp4"))
                    if scene_videos:
                        return sorted(scene_videos)[0]

        logger.warning(f"No video found for participant: {participant}")
        return None

    def get_participant_signals_dir(self, participant: str) -> dict[str, Optional[Path]]:
        """Find signal data directories for a participant.

        Args:
            participant: Participant directory name

        Returns:
            Dictionary with paths to gaze, hr, and eda directories/files
        """
        participant_dir = self.data_root / participant
        signals = {"gaze": None, "hr": None, "eda": None}

        # Find gaze data in Timeseries directory
        timeseries_dir = participant_dir / "Timeseries Data + Scene Video"
        if timeseries_dir.exists():
            for session_dir in timeseries_dir.iterdir():
                if session_dir.is_dir():
                    gaze_file = session_dir / "gaze.csv"
                    if gaze_file.exists():
                        signals["gaze"] = gaze_file
                        break

        # Find gaze data in raw_data/session/export/
        raw_data_dir = participant_dir / "raw_data"
        if not signals["gaze"] and raw_data_dir.exists():
            for session_dir in raw_data_dir.iterdir():
                if session_dir.is_dir() and not session_dir.name.startswith("Watch_"):
                    gaze_file = session_dir / "export" / "gaze.csv"
                    if gaze_file.exists():
                        signals["gaze"] = gaze_file
                        break

        # Find HR/EDA in Watch directories
        raw_data_dir = participant_dir / "raw_data"
        if raw_data_dir.exists():
            for watch_dir in raw_data_dir.glob("Watch_*"):
                if watch_dir.is_dir():
                    hr_file = watch_dir / "hr.csv"
                    eda_file = watch_dir / "eda.csv"
                    if hr_file.exists():
                        signals["hr"] = hr_file
                    if eda_file.exists():
                        signals["eda"] = eda_file
                    if signals["hr"] and signals["eda"]:
                        break

        return signals

    def get_statistics(self) -> dict:
        """Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "participants": [],
            "task_2_1": {"participants": [], "total_entries": 0},
            "task_2_2": {"participants": [], "total_entries": 0},
        }

        participants = self.discover_participants()
        stats["participants"] = participants

        for participant in participants:
            entries_2_1 = self.load_task_2_1_annotations(participant)
            if entries_2_1:
                stats["task_2_1"]["participants"].append(participant)
                stats["task_2_1"]["total_entries"] += len(entries_2_1)

            entries_2_2 = self.load_task_2_2_annotations(participant)
            if entries_2_2:
                stats["task_2_2"]["participants"].append(participant)
                stats["task_2_2"]["total_entries"] += len(entries_2_2)

        return stats
