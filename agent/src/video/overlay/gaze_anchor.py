"""Gaze anchor extraction for UI positioning.

Extracts gaze coordinates from eye-tracking CSV files to determine
where UI overlays should be positioned on video frames.
"""

import csv
import logging
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

FallbackStrategy = Literal["last_valid", "center", "interpolate"]


class GazeAnchorExtractor:
    """Extract gaze anchor points from eye-tracking data.

    Loads gaze data from CSV files and provides methods to query
    gaze coordinates at specific timestamps.

    Attributes:
        gaze_data: Numpy array of [time_s, gaze_x, gaze_y] values.
        valid_mask: Boolean mask indicating rows with valid gaze data.
    """

    def __init__(self, gaze_csv_path: Path | str):
        """Initialize the gaze anchor extractor.

        Args:
            gaze_csv_path: Path to the gaze.csv file.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the CSV format is invalid.
        """
        self.gaze_csv_path = Path(gaze_csv_path)

        if not self.gaze_csv_path.exists():
            raise FileNotFoundError(f"Gaze CSV not found: {self.gaze_csv_path}")

        self._load_gaze_data()

    def _load_gaze_data(self) -> None:
        """Load and parse the gaze CSV file."""
        times = []
        gaze_x = []
        gaze_y = []

        with open(self.gaze_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate expected columns
            expected_cols = {"time_s", "gaze x [px]", "gaze y [px]"}
            if not expected_cols.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"Invalid CSV format. Expected columns: {expected_cols}, "
                    f"got: {reader.fieldnames}"
                )

            for row in reader:
                try:
                    time_val = float(row["time_s"])
                    # Handle empty gaze values (marked as NaN)
                    x_str = row["gaze x [px]"].strip()
                    y_str = row["gaze y [px]"].strip()

                    x_val = float(x_str) if x_str else np.nan
                    y_val = float(y_str) if y_str else np.nan

                    times.append(time_val)
                    gaze_x.append(x_val)
                    gaze_y.append(y_val)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid row: {e}")
                    continue

        if not times:
            raise ValueError(f"No valid gaze data found in {self.gaze_csv_path}")

        # Store as numpy arrays for efficient querying
        self.gaze_data = np.column_stack([times, gaze_x, gaze_y])
        self.valid_mask = ~np.isnan(self.gaze_data[:, 1]) & ~np.isnan(self.gaze_data[:, 2])

        valid_count = np.sum(self.valid_mask)
        total_count = len(self.valid_mask)
        logger.info(
            f"Loaded gaze data: {total_count} samples, "
            f"{valid_count} valid ({100*valid_count/total_count:.1f}%)"
        )

    def get_anchor_at_time(
        self,
        timestamp: float,
        fallback: FallbackStrategy = "last_valid",
        default_center: tuple[float, float] = (640.0, 360.0),
    ) -> tuple[float, float] | None:
        """Get gaze anchor coordinates at a specific timestamp.

        Args:
            timestamp: The timestamp in seconds to query.
            fallback: Strategy for handling missing gaze data:
                - "last_valid": Use the most recent valid gaze position
                - "center": Return the default center coordinates
                - "interpolate": Interpolate between nearest valid points
            default_center: Default center coordinates for "center" fallback.

        Returns:
            Tuple of (x, y) gaze coordinates, or None if no valid data.
        """
        if len(self.gaze_data) == 0:
            return None

        times = self.gaze_data[:, 0]

        # Find the closest timestamp
        idx = np.searchsorted(times, timestamp)

        # Clamp to valid range
        if idx >= len(times):
            idx = len(times) - 1
        elif idx > 0:
            # Choose the closer of the two adjacent timestamps
            if abs(times[idx] - timestamp) > abs(times[idx - 1] - timestamp):
                idx = idx - 1

        # Check if we have valid gaze at this index
        if self.valid_mask[idx]:
            return (self.gaze_data[idx, 1], self.gaze_data[idx, 2])

        # Apply fallback strategy
        if fallback == "center":
            logger.debug(f"No valid gaze at {timestamp:.3f}s, using center fallback")
            return default_center

        elif fallback == "last_valid":
            # Search backwards for the last valid point
            valid_before = np.where(self.valid_mask[:idx])[0]
            if len(valid_before) > 0:
                last_valid_idx = valid_before[-1]
                logger.debug(
                    f"No valid gaze at {timestamp:.3f}s, using last valid at "
                    f"{times[last_valid_idx]:.3f}s"
                )
                return (
                    self.gaze_data[last_valid_idx, 1],
                    self.gaze_data[last_valid_idx, 2],
                )
            # Fall through to center if no previous valid point
            logger.debug(f"No valid gaze before {timestamp:.3f}s, using center fallback")
            return default_center

        elif fallback == "interpolate":
            # Find nearest valid points before and after
            valid_before = np.where(self.valid_mask[:idx])[0]
            valid_after = np.where(self.valid_mask[idx:])[0] + idx

            if len(valid_before) > 0 and len(valid_after) > 0:
                before_idx = valid_before[-1]
                after_idx = valid_after[0]

                # Linear interpolation
                t_before = times[before_idx]
                t_after = times[after_idx]
                t_range = t_after - t_before

                if t_range > 0:
                    alpha = (timestamp - t_before) / t_range
                    x = (1 - alpha) * self.gaze_data[before_idx, 1] + alpha * self.gaze_data[after_idx, 1]
                    y = (1 - alpha) * self.gaze_data[before_idx, 2] + alpha * self.gaze_data[after_idx, 2]
                    logger.debug(f"Interpolated gaze at {timestamp:.3f}s")
                    return (x, y)

            # Fall back to last_valid if interpolation not possible
            return self.get_anchor_at_time(timestamp, fallback="last_valid", default_center=default_center)

        return None

    def get_time_range(self) -> tuple[float, float]:
        """Get the time range covered by the gaze data.

        Returns:
            Tuple of (start_time, end_time) in seconds.
        """
        if len(self.gaze_data) == 0:
            return (0.0, 0.0)
        return (self.gaze_data[0, 0], self.gaze_data[-1, 0])

    def get_valid_ratio(self) -> float:
        """Get the ratio of valid gaze samples.

        Returns:
            Ratio of valid samples (0.0 to 1.0).
        """
        if len(self.valid_mask) == 0:
            return 0.0
        return np.sum(self.valid_mask) / len(self.valid_mask)
