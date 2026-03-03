"""
Unit tests for pineapple_health_detector.py
Run with: pytest tests/
"""

import csv
import sys
import os
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the parent directory importable without installing as a package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub out picamera2 / picamera so the module can be imported in CI (no RPi)
for _mod in ("picamera2", "picamera"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

import pineapple_health_detector as phd  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def healthy_frame():
    """BGR frame filled with a healthy green colour (H≈55, S≈200, V≈180)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 160, 50)   # BGR ≈ vibrant green
    return frame


@pytest.fixture()
def yellowing_frame():
    """BGR frame with a yellow-green colour indicating nutrient stress."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (20, 200, 200)  # BGR ≈ yellow
    return frame


@pytest.fixture()
def brown_frame():
    """BGR frame with brown colouring indicating disease / necrosis."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (20, 60, 120)   # BGR ≈ brown
    return frame


@pytest.fixture()
def black_frame():
    """BGR frame with no plant pixels (bare soil / shadow)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# preprocess()
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_returns_same_shape(self, healthy_frame):
        result = phd.preprocess(healthy_frame)
        assert result.shape == healthy_frame.shape

    def test_returns_uint8(self, healthy_frame):
        result = phd.preprocess(healthy_frame)
        assert result.dtype == np.uint8

    def test_does_not_return_identical_array(self, healthy_frame):
        # Preprocessing must alter the array in some way
        result = phd.preprocess(healthy_frame)
        # At minimum the objects should be different
        assert result is not healthy_frame


# ---------------------------------------------------------------------------
# segment_plant_regions()
# ---------------------------------------------------------------------------

class TestSegmentPlantRegions:
    def test_green_frame_has_plant_pixels(self, healthy_frame):
        mask = phd.segment_plant_regions(healthy_frame)
        assert np.count_nonzero(mask) > 0

    def test_black_frame_has_no_plant_pixels(self, black_frame):
        mask = phd.segment_plant_regions(black_frame)
        assert np.count_nonzero(mask) == 0

    def test_mask_is_binary(self, healthy_frame):
        mask = phd.segment_plant_regions(healthy_frame)
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255})

    def test_mask_shape_matches_input(self, healthy_frame):
        mask = phd.segment_plant_regions(healthy_frame)
        assert mask.shape == healthy_frame.shape[:2]


# ---------------------------------------------------------------------------
# classify_health()
# ---------------------------------------------------------------------------

class TestClassifyHealth:
    def _full_mask(self, frame):
        return np.full(frame.shape[:2], 255, dtype=np.uint8)

    def test_healthy_frame_classified_as_healthy(self, healthy_frame):
        mask    = self._full_mask(healthy_frame)
        metrics = phd.classify_health(healthy_frame, mask)
        assert metrics["status"] == phd.STATUS_HEALTHY

    def test_no_plant_pixels_returns_no_plant_detected(self, healthy_frame):
        empty_mask = np.zeros(healthy_frame.shape[:2], dtype=np.uint8)
        metrics    = phd.classify_health(healthy_frame, empty_mask)
        assert metrics["status"] == "No plant detected"
        assert metrics["total_plant_pixels"] == 0

    def test_metrics_keys_present(self, healthy_frame):
        mask    = self._full_mask(healthy_frame)
        metrics = phd.classify_health(healthy_frame, mask)
        expected_keys = {
            "total_plant_pixels",
            "healthy_px", "yellowing_px", "brown_px",
            "healthy_pct", "yellowing_pct", "brown_pct",
            "status",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_percentages_non_negative(self, healthy_frame):
        mask    = self._full_mask(healthy_frame)
        metrics = phd.classify_health(healthy_frame, mask)
        assert metrics["healthy_pct"]   >= 0
        assert metrics["yellowing_pct"] >= 0
        assert metrics["brown_pct"]     >= 0

    def test_total_plant_pixels_matches_mask(self, healthy_frame):
        mask    = self._full_mask(healthy_frame)
        metrics = phd.classify_health(healthy_frame, mask)
        assert metrics["total_plant_pixels"] == int(np.count_nonzero(mask))


# ---------------------------------------------------------------------------
# detect_plant_contours()
# ---------------------------------------------------------------------------

class TestDetectPlantContours:
    def test_returns_list(self, healthy_frame):
        mask     = phd.segment_plant_regions(healthy_frame)
        contours = phd.detect_plant_contours(mask)
        assert isinstance(contours, list)

    def test_sorted_largest_first(self, healthy_frame):
        mask     = phd.segment_plant_regions(healthy_frame)
        contours = phd.detect_plant_contours(mask)
        if len(contours) > 1:
            areas = [cv2.contourArea(c) for c in contours]
            assert areas == sorted(areas, reverse=True)

    def test_empty_mask_returns_empty_list(self, black_frame):
        mask     = phd.segment_plant_regions(black_frame)
        contours = phd.detect_plant_contours(mask)
        assert contours == []


# ---------------------------------------------------------------------------
# annotate_frame()
# ---------------------------------------------------------------------------

class TestAnnotateFrame:
    def _metrics(self, status=phd.STATUS_HEALTHY):
        return {
            "status":          status,
            "healthy_pct":     70.0,
            "yellowing_pct":   20.0,
            "brown_pct":       10.0,
            "total_plant_pixels": 10000,
        }

    def test_annotated_same_shape(self, healthy_frame):
        annotated = phd.annotate_frame(healthy_frame, self._metrics(), [])
        assert annotated.shape == healthy_frame.shape

    def test_annotated_is_copy(self, healthy_frame):
        annotated = phd.annotate_frame(healthy_frame, self._metrics(), [])
        assert annotated is not healthy_frame

    @pytest.mark.parametrize("status", [
        phd.STATUS_HEALTHY,
        phd.STATUS_MODERATE,
        phd.STATUS_SEVERE,
    ])
    def test_all_statuses_do_not_raise(self, healthy_frame, status):
        phd.annotate_frame(healthy_frame, self._metrics(status), [])


# ---------------------------------------------------------------------------
# process_frame()
# ---------------------------------------------------------------------------

class TestProcessFrame:
    def test_returns_four_values(self, healthy_frame):
        result = phd.process_frame(healthy_frame)
        assert len(result) == 4

    def test_metrics_is_dict(self, healthy_frame):
        metrics, _, _, _ = phd.process_frame(healthy_frame)
        assert isinstance(metrics, dict)

    def test_annotated_shape_matches_input(self, healthy_frame):
        _, _, annotated, _ = phd.process_frame(healthy_frame)
        assert annotated.shape == healthy_frame.shape


# ---------------------------------------------------------------------------
# save_results()  — filesystem integration
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_csv_created_with_header(self, tmp_path, healthy_frame):
        metrics = {
            "status":             phd.STATUS_HEALTHY,
            "healthy_pct":        75.0,
            "yellowing_pct":      15.0,
            "brown_pct":          10.0,
            "total_plant_pixels": 50000,
        }
        annotated = healthy_frame.copy()
        phd.save_results(
            tmp_path, "20260101_120000",
            healthy_frame, annotated, metrics, "health_log.csv"
        )
        csv_path = tmp_path / "health_log.csv"
        assert csv_path.exists()
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows   = list(reader)
        assert len(rows) == 1
        assert rows[0]["status"] == phd.STATUS_HEALTHY

    def test_raw_image_saved(self, tmp_path, healthy_frame):
        metrics = {
            "status": "No plant detected",
            "healthy_pct": 0.0, "yellowing_pct": 0.0, "brown_pct": 0.0,
            "total_plant_pixels": 0,
        }
        phd.save_results(
            tmp_path, "20260101_120001",
            healthy_frame, healthy_frame, metrics, "health_log.csv"
        )
        assert (tmp_path / "20260101_120001_raw.jpg").exists()

    def test_csv_appends_multiple_rows(self, tmp_path, healthy_frame):
        metrics = {
            "status":             phd.STATUS_MODERATE,
            "healthy_pct":        40.0,
            "yellowing_pct":      40.0,
            "brown_pct":          20.0,
            "total_plant_pixels": 30000,
        }
        for ts in ("20260101_120002", "20260101_120003"):
            phd.save_results(
                tmp_path, ts,
                healthy_frame, healthy_frame, metrics, "health_log.csv"
            )
        with open(tmp_path / "health_log.csv") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# analyze_image_file()
# ---------------------------------------------------------------------------

class TestAnalyzeImageFile:
    def test_analyze_existing_file(self, tmp_path, healthy_frame):
        img_path = str(tmp_path / "test_plant.jpg")
        cv2.imwrite(img_path, healthy_frame)
        metrics = phd.analyze_image_file(img_path)
        assert "status" in metrics

    def test_analyze_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            phd.analyze_image_file("/nonexistent/path/image.jpg")
