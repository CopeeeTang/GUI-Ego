"""Signal reader — parse gaze, heart rate, and EDA CSV files."""

import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SignalReader:
    """Parse physiological signal CSV files from a sample directory.

    Expected files:
        signals/gaze.csv — columns: time_s, gaze x [px], gaze y [px], worn, fixation id
        signals/hr.csv   — columns: time_s, hr
        signals/eda.csv  — columns: time_s, eda
    """

    def __init__(self, signals_dir: str | Path):
        self.signals_dir = Path(signals_dir)
        self.gaze_data: list[dict] = []
        self.hr_data: list[dict] = []
        self.eda_data: list[dict] = []

        self._load_all()

    def _load_all(self):
        """Load all available signal files."""
        self.gaze_data = self._load_csv("gaze.csv", self._parse_gaze_row)
        self.hr_data = self._load_csv("hr.csv", self._parse_hr_row)
        self.eda_data = self._load_csv("eda.csv", self._parse_eda_row)

        logger.info(
            f"Loaded signals — gaze: {len(self.gaze_data)}, "
            f"hr: {len(self.hr_data)}, eda: {len(self.eda_data)}"
        )

    def _load_csv(self, filename: str, row_parser) -> list[dict]:
        """Load a CSV file, gracefully handling missing files."""
        filepath = self.signals_dir / filename
        if not filepath.exists():
            logger.warning(f"Signal file not found: {filepath}")
            return []

        rows = []
        try:
            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed = row_parser(row)
                    if parsed is not None:
                        rows.append(parsed)
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")

        return rows

    def _parse_gaze_row(self, row: dict) -> Optional[dict]:
        """Parse a gaze CSV row."""
        try:
            return {
                "time_s": float(row["time_s"]),
                "gaze_x": float(row.get("gaze x [px]", 0)),
                "gaze_y": float(row.get("gaze y [px]", 0)),
                "worn": row.get("worn", "True").strip().lower() == "true",
                "fixation_id": row.get("fixation id", "").strip() or None,
            }
        except (ValueError, KeyError) as e:
            logger.debug(f"Skipping gaze row: {e}")
            return None

    def _parse_hr_row(self, row: dict) -> Optional[dict]:
        """Parse a heart rate CSV row."""
        try:
            return {
                "time_s": float(row["time_s"]),
                "hr": float(row["hr"]),
            }
        except (ValueError, KeyError) as e:
            logger.debug(f"Skipping HR row: {e}")
            return None

    def _parse_eda_row(self, row: dict) -> Optional[dict]:
        """Parse an EDA CSV row."""
        try:
            return {
                "time_s": float(row["time_s"]),
                "eda": float(row["eda"]),
            }
        except (ValueError, KeyError) as e:
            logger.debug(f"Skipping EDA row: {e}")
            return None

    def get_gaze_at(self, timestamp: float, window_sec: float = 0.25) -> list[dict]:
        """Get gaze samples near a timestamp."""
        return [
            g for g in self.gaze_data
            if abs(g["time_s"] - timestamp) <= window_sec
        ]

    def get_hr_at(self, timestamp: float, window_sec: float = 1.0) -> list[dict]:
        """Get HR samples near a timestamp."""
        return [
            h for h in self.hr_data
            if abs(h["time_s"] - timestamp) <= window_sec
        ]

    def get_eda_at(self, timestamp: float, window_sec: float = 0.5) -> list[dict]:
        """Get EDA samples near a timestamp."""
        return [
            e for e in self.eda_data
            if abs(e["time_s"] - timestamp) <= window_sec
        ]

    @property
    def time_range(self) -> tuple[float, float]:
        """Get the overall time range covered by signals."""
        all_times = (
            [g["time_s"] for g in self.gaze_data]
            + [h["time_s"] for h in self.hr_data]
            + [e["time_s"] for e in self.eda_data]
        )
        if not all_times:
            return (0.0, 0.0)
        return (min(all_times), max(all_times))
