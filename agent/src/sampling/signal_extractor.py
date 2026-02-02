"""Signal extraction module for extracting gaze, HR, and EDA data."""

import csv
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SignalExtractor:
    """Extract physiological signals from Ego-Dataset recordings."""

    def __init__(self, signals_config: dict[str, Optional[Path]]):
        """Initialize signal extractor.

        Args:
            signals_config: Dictionary with paths to signal files
                           {"gaze": Path, "hr": Path, "eda": Path}
        """
        self.signals_config = signals_config
        self._gaze_df: Optional[pd.DataFrame] = None
        self._hr_df: Optional[pd.DataFrame] = None
        self._eda_df: Optional[pd.DataFrame] = None
        self._gaze_start_ns: Optional[int] = None

    def _load_gaze_data(self) -> Optional[pd.DataFrame]:
        """Load and cache gaze data."""
        if self._gaze_df is not None:
            return self._gaze_df

        gaze_path = self.signals_config.get("gaze")
        if gaze_path is None or not gaze_path.exists():
            logger.warning("Gaze data file not found")
            return None

        try:
            self._gaze_df = pd.read_csv(gaze_path)
            # Store the first timestamp as reference for relative time conversion
            if "timestamp [ns]" in self._gaze_df.columns and len(self._gaze_df) > 0:
                self._gaze_start_ns = self._gaze_df["timestamp [ns]"].iloc[0]
            logger.debug(f"Loaded {len(self._gaze_df)} gaze samples")
            return self._gaze_df
        except Exception as e:
            logger.error(f"Error loading gaze data: {e}")
            return None

    def _load_hr_data(self) -> Optional[pd.DataFrame]:
        """Load and cache heart rate data."""
        if self._hr_df is not None:
            return self._hr_df

        hr_path = self.signals_config.get("hr")
        if hr_path is None or not hr_path.exists():
            logger.warning("HR data file not found")
            return None

        try:
            self._hr_df = pd.read_csv(hr_path)
            logger.debug(f"Loaded {len(self._hr_df)} HR samples")
            return self._hr_df
        except Exception as e:
            logger.error(f"Error loading HR data: {e}")
            return None

    def _load_eda_data(self) -> Optional[pd.DataFrame]:
        """Load and cache EDA data."""
        if self._eda_df is not None:
            return self._eda_df

        eda_path = self.signals_config.get("eda")
        if eda_path is None or not eda_path.exists():
            logger.warning("EDA data file not found")
            return None

        try:
            self._eda_df = pd.read_csv(eda_path)
            logger.debug(f"Loaded {len(self._eda_df)} EDA samples")
            return self._eda_df
        except Exception as e:
            logger.error(f"Error loading EDA data: {e}")
            return None

    def extract_gaze(
        self,
        start_time: float,
        end_time: float,
        window: float = 5.0,
    ) -> dict:
        """Extract gaze data for a time window.

        Args:
            start_time: Start time in seconds (from video start)
            end_time: End time in seconds (from video start)
            window: Additional seconds to include before/after (default ±5s)

        Returns:
            Dictionary with gaze data and metadata
        """
        df = self._load_gaze_data()
        if df is None:
            return {"available": False, "data": [], "window": f"±{window}s"}

        # Calculate window boundaries
        window_start = max(0, start_time - window)
        window_end = end_time + window

        # Convert seconds to nanoseconds relative to start
        if self._gaze_start_ns is not None and "timestamp [ns]" in df.columns:
            window_start_ns = self._gaze_start_ns + int(window_start * 1e9)
            window_end_ns = self._gaze_start_ns + int(window_end * 1e9)

            # Filter data
            mask = (df["timestamp [ns]"] >= window_start_ns) & (df["timestamp [ns]"] <= window_end_ns)
            filtered = df[mask].copy()

            # Convert timestamp to relative seconds
            filtered["time_s"] = (filtered["timestamp [ns]"] - self._gaze_start_ns) / 1e9

            # Select relevant columns
            columns = ["time_s", "gaze x [px]", "gaze y [px]", "worn"]
            if "fixation id" in filtered.columns:
                columns.append("fixation id")

            result_data = filtered[columns].to_dict(orient="records")

            return {
                "available": True,
                "data": result_data,
                "sample_count": len(result_data),
                "window": f"±{window}s",
                "time_range": {"start": window_start, "end": window_end},
            }
        else:
            return {"available": False, "data": [], "window": f"±{window}s"}

    def extract_hr(
        self,
        start_time: float,
        end_time: float,
        window: float = 5.0,
    ) -> dict:
        """Extract heart rate data for a time window.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            window: Additional seconds to include before/after (default ±5s)

        Returns:
            Dictionary with HR data and metadata
        """
        df = self._load_hr_data()
        if df is None:
            return {"available": False, "data": [], "window": f"±{window}s"}

        # Calculate window boundaries in milliseconds
        window_start_ms = max(0, (start_time - window) * 1000)
        window_end_ms = (end_time + window) * 1000

        # Filter data (timestamp is in milliseconds from recording start)
        if "timestamp" in df.columns:
            mask = (df["timestamp"] >= window_start_ms) & (df["timestamp"] <= window_end_ms)
            filtered = df[mask].copy()

            # Convert timestamp to seconds
            filtered["time_s"] = filtered["timestamp"] / 1000.0

            result_data = filtered[["time_s", "hr"]].to_dict(orient="records")

            return {
                "available": True,
                "data": result_data,
                "sample_count": len(result_data),
                "window": f"±{window}s",
                "time_range": {"start": max(0, start_time - window), "end": end_time + window},
            }
        else:
            return {"available": False, "data": [], "window": f"±{window}s"}

    def extract_eda(
        self,
        start_time: float,
        end_time: float,
        window: float = 5.0,
    ) -> dict:
        """Extract EDA (electrodermal activity) data for a time window.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            window: Additional seconds to include before/after (default ±5s)

        Returns:
            Dictionary with EDA data and metadata
        """
        df = self._load_eda_data()
        if df is None:
            return {"available": False, "data": [], "window": f"±{window}s"}

        # Calculate window boundaries in milliseconds
        window_start_ms = max(0, (start_time - window) * 1000)
        window_end_ms = (end_time + window) * 1000

        # Filter data (timestamp is in milliseconds from recording start)
        if "timestamp" in df.columns:
            mask = (df["timestamp"] >= window_start_ms) & (df["timestamp"] <= window_end_ms)
            filtered = df[mask].copy()

            # Convert timestamp to seconds
            filtered["time_s"] = filtered["timestamp"] / 1000.0

            result_data = filtered[["time_s", "eda"]].to_dict(orient="records")

            return {
                "available": True,
                "data": result_data,
                "sample_count": len(result_data),
                "window": f"±{window}s",
                "time_range": {"start": max(0, start_time - window), "end": end_time + window},
            }
        else:
            return {"available": False, "data": [], "window": f"±{window}s"}

    def extract_all_signals(
        self,
        start_time: float,
        end_time: float,
        window: float = 5.0,
    ) -> dict:
        """Extract all available signals for a time window.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            window: Additional seconds to include before/after (default ±5s)

        Returns:
            Dictionary with all signal data
        """
        return {
            "gaze": self.extract_gaze(start_time, end_time, window),
            "heart_rate": self.extract_hr(start_time, end_time, window),
            "eda": self.extract_eda(start_time, end_time, window),
        }

    def save_signals_to_csv(
        self,
        start_time: float,
        end_time: float,
        output_dir: Path,
        window: float = 5.0,
    ) -> dict[str, Optional[Path]]:
        """Extract and save signals to CSV files.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            output_dir: Directory to save CSV files
            window: Additional seconds to include before/after (default ±5s)

        Returns:
            Dictionary with paths to saved CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {"gaze": None, "hr": None, "eda": None}

        # Extract and save gaze
        gaze_data = self.extract_gaze(start_time, end_time, window)
        if gaze_data["available"] and gaze_data["data"]:
            gaze_path = output_dir / "gaze.csv"
            df = pd.DataFrame(gaze_data["data"])
            df.to_csv(gaze_path, index=False)
            saved_paths["gaze"] = gaze_path
            logger.debug(f"Saved gaze data to {gaze_path}")

        # Extract and save HR
        hr_data = self.extract_hr(start_time, end_time, window)
        if hr_data["available"] and hr_data["data"]:
            hr_path = output_dir / "hr.csv"
            df = pd.DataFrame(hr_data["data"])
            df.to_csv(hr_path, index=False)
            saved_paths["hr"] = hr_path
            logger.debug(f"Saved HR data to {hr_path}")

        # Extract and save EDA
        eda_data = self.extract_eda(start_time, end_time, window)
        if eda_data["available"] and eda_data["data"]:
            eda_path = output_dir / "eda.csv"
            df = pd.DataFrame(eda_data["data"])
            df.to_csv(eda_path, index=False)
            saved_paths["eda"] = eda_path
            logger.debug(f"Saved EDA data to {eda_path}")

        return saved_paths
