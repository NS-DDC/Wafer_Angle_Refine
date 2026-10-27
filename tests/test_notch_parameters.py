"""Independent public-parameter contract and synthetic option integration tests.

The expected values are a fixed snapshot of the V7 public notch API. Tests do
not import, locate, or otherwise depend on an upstream checkout.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULE = importlib.import_module("angle")
angle = MODULE.angle


EXPECTED_DEFAULTS = {
    "notch_reference_angle_deg": 90.0,
    "notch_max_dimension": 3072,
    "notch_angle_samples": 3600,
    "notch_baseline_window_deg": 10.0,
    "notch_min_depth_px": None,
    "notch_min_depth_ratio": 0.001,
    "notch_min_wide_deg": 2.0,
    "notch_search_center_angle_deg": 90.0,
    "notch_search_half_width_deg": 45.0,
    "notch_wafer_center_hint_px": None,
    "notch_wafer_radius_hint_px": None,
    "notch_roi_center_px": None,
    "notch_roi_half_size_px": 600.0,
    "notch_semicircle_radius_range_px": None,
    "notch_semicircle_min_score": 0.55,
    "notch_use_roi_background": True,
    "notch_background_palette_size": 3,
    "notch_background_outer_band_fraction": 0.28,
    "notch_background_distance_threshold_lab": None,
    "notch_background_noise_margin_lab": 4.0,
    "notch_background_morph_px": 24.0,
    "notch_fallback_mode": "rim_intrusion",
    "notch_failure_mode": "error",
    "alignment_interpolation": cv2.INTER_CUBIC,
    "alignment_border_value": (0, 0, 0),
}

DETECTOR_NAMES = {
    "notch_reference_angle_deg": "reference_angle_deg",
    "notch_max_dimension": "max_dimension",
    "notch_angle_samples": "angle_samples",
    "notch_baseline_window_deg": "baseline_window_deg",
    "notch_min_depth_px": "min_notch_depth_px",
    "notch_min_depth_ratio": "min_notch_depth_ratio",
    "notch_min_wide_deg": "min_wide_notch_deg",
    "notch_search_center_angle_deg": "search_center_angle_deg",
    "notch_search_half_width_deg": "search_half_width_deg",
    "notch_wafer_center_hint_px": "wafer_center_hint_px",
    "notch_wafer_radius_hint_px": "wafer_radius_hint_px",
    "notch_roi_center_px": "notch_roi_center_px",
    "notch_roi_half_size_px": "notch_roi_half_size_px",
    "notch_semicircle_radius_range_px": "notch_semicircle_radius_range_px",
    "notch_semicircle_min_score": "notch_semicircle_min_score",
    "notch_use_roi_background": "notch_use_roi_background",
    "notch_background_palette_size": "notch_background_palette_size",
    "notch_background_outer_band_fraction": "notch_background_outer_band_fraction",
    "notch_background_distance_threshold_lab": "notch_background_distance_threshold_lab",
    "notch_background_noise_margin_lab": "notch_background_noise_margin_lab",
    "notch_background_morph_px": "notch_background_morph_px",
    "notch_fallback_mode": "notch_fallback_mode",
    "notch_failure_mode": "failure_mode",
}

CUSTOM_OPTIONS = {
    "notch_reference_angle_deg": 180.0,
    "notch_max_dimension": 2048,
    "notch_angle_samples": 2400,
    "notch_baseline_window_deg": 12.0,
    "notch_min_depth_px": 3.5,
    "notch_min_depth_ratio": 0.002,
    "notch_min_wide_deg": 1.5,
    "notch_search_center_angle_deg": 175.0,
    "notch_search_half_width_deg": 30.0,
    "notch_wafer_center_hint_px": (4.0, 3.0),
    "notch_wafer_radius_hint_px": 2.5,
    "notch_roi_center_px": (1.5, 3.0),
    "notch_roi_half_size_px": (2.0, 1.5),
    "notch_semicircle_radius_range_px": (0.5, 1.5),
    "notch_semicircle_min_score": 0.60,
    "notch_use_roi_background": False,
    "notch_background_palette_size": 5,
    "notch_background_outer_band_fraction": 0.35,
    "notch_background_distance_threshold_lab": 11.0,
    "notch_background_noise_margin_lab": 6.0,
    "notch_background_morph_px": 5.0,
    "notch_fallback_mode": "none",
    "notch_failure_mode": "zero",
    "alignment_interpolation": cv2.INTER_NEAREST,
    "alignment_border_value": (12, 34, 56, 78),
}


def fixture(notch="triangle"):
    image = np.zeros((900, 900, 3), dtype=np.uint8)
    cv2.circle(image, (450, 450), 390, (110, 130, 145), -1, cv2.LINE_AA)
    cv2.circle(image, (450, 450), 390, (185, 195, 205), 5, cv2.LINE_AA)
    if notch == "triangle":
        cv2.fillPoly(
            image, [np.asarray(((428, 864), (450, 825), (472, 864)), np.int32)],
            (0, 0, 0), cv2.LINE_AA,
        )
    elif notch == "semicircle":
        cv2.circle(image, (450, 840), 30, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.rectangle(image, (410, 840), (490, 899), (0, 0, 0), -1)
    elif notch != "none":
        raise ValueError(notch)
    cv2.rectangle(image, (590, 440), (610, 460), (0, 0, 255), -1)
    return image


def rotate(image, degrees):
    return cv2.warpAffine(
        image, cv2.getRotationMatrix2D((450.0, 450.0), degrees, 1.0),
        (900, 900), flags=cv2.INTER_CUBIC,
    )


def marker_angle(image):
    mask = (image[..., 2] > 225) & (image[..., 0] < 40) & (image[..., 1] < 40)
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise AssertionError("Independent red landmark disappeared.")
    return float(np.degrees(np.arctan2(ys.mean() - 450, xs.mean() - 450)))


class NotchParameterTests(unittest.TestCase):
    def test_exact_keyword_names_and_defaults_match_v7_snapshot(self):
        signature = inspect.signature(angle)
        actual = {name: value.default for name, value in signature.parameters.items() if name != "src"}
        self.assertEqual(actual, EXPECTED_DEFAULTS)
        for name in EXPECTED_DEFAULTS:
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_every_detector_option_reaches_its_original_argument(self):
        source = np.zeros((7, 9, 4), dtype=np.uint8)
        source[..., 3] = 255
        for options in ({}, CUSTOM_OPTIONS):
            with self.subTest(custom=bool(options)):
                result = SimpleNamespace(wafer_center_px=(4.0, 3.0), correction_angle_deg=0.0)
                with mock.patch.object(MODULE, "_detect_wafer_notch", return_value=result) as detector:
                    output = angle(source, **options)
                detector.assert_called_once()
                np.testing.assert_array_equal(detector.call_args.args[0], source[..., :3])
                values = EXPECTED_DEFAULTS | options
                expected = {internal: values[public] for public, internal in DETECTOR_NAMES.items()}
                self.assertEqual(detector.call_args.kwargs, expected)
                self.assertEqual(output.shape, source.shape)
                self.assertFalse(np.shares_memory(source, output))

    def test_actual_roi_semicircle_and_geometry_hints_align_known_landmark(self):
        source = rotate(fixture("semicircle"), 13.0)
        before = source.copy()
        rotation = cv2.getRotationMatrix2D((450.0, 450.0), 13.0, 1.0)
        roi = tuple(rotation @ np.asarray((450.0, 840.0, 1.0)))
        with mock.patch.object(
            MODULE, "_learn_background_from_notch_roi",
            wraps=MODULE._learn_background_from_notch_roi,
        ) as roi_analysis:
            output = angle(
                source, notch_roi_center_px=roi, notch_roi_half_size_px=(110, 90),
                notch_semicircle_radius_range_px=(15, 45),
                notch_wafer_center_hint_px=(450.0, 450.0), notch_wafer_radius_hint_px=390.0,
            )
        roi_analysis.assert_called_once()
        self.assertLess(abs(marker_angle(output)), 0.5)
        np.testing.assert_array_equal(source, before)
        self.assertFalse(np.shares_memory(source, output))

    def test_nonbottom_search_and_reference_change_the_actual_target(self):
        # Rotating +70 puts the original bottom notch at image angle 20.
        # Reference 0 should rotate it +20 to the right, and the marker up.
        source = rotate(fixture(), 70.0)
        with self.assertRaises(RuntimeError):
            angle(source)
        output = angle(
            source, notch_search_center_angle_deg=20.0,
            notch_search_half_width_deg=20.0, notch_reference_angle_deg=0.0,
        )
        self.assertLess(abs(marker_angle(output) + 90.0), 0.5)
        expected = rotate(source, 20.0)
        error = np.mean(np.abs(output.astype(np.float32) - expected.astype(np.float32)))
        self.assertLess(float(error), 1.5)

    def test_zero_failure_policy_keeps_no_notch_image_in_all_channel_formats(self):
        bgr = fixture("none")
        inputs = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA))
        for source in inputs:
            with self.subTest(shape=source.shape):
                before = source.copy()
                output = angle(source, notch_failure_mode="zero")
                np.testing.assert_array_equal(output, source)
                np.testing.assert_array_equal(source, before)
                self.assertEqual(output.dtype, np.uint8)
                self.assertFalse(np.shares_memory(source, output))

    def test_zero_notch_policy_does_not_hide_missing_wafer_edge(self):
        with self.assertRaises(RuntimeError):
            angle(np.zeros((400, 400, 3), np.uint8), notch_failure_mode="zero")

    def test_custom_interpolation_and_four_channel_border_are_applied(self):
        source = np.arange(9 * 11 * 4, dtype=np.uint16).reshape(9, 11, 4).astype(np.uint8)
        result = SimpleNamespace(wafer_center_px=(5.0, 4.0), correction_angle_deg=17.25)
        border = (12, 34, 56, 78)
        with mock.patch.object(MODULE, "_detect_wafer_notch", return_value=result):
            output = angle(source, alignment_interpolation=cv2.INTER_NEAREST, alignment_border_value=border)
        matrix = cv2.getRotationMatrix2D((5.0, 4.0), 17.25, 1.0)
        expected = cv2.warpAffine(
            source, matrix, (11, 9), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=border,
        )
        np.testing.assert_array_equal(output, expected)
        self.assertTrue(np.any(np.all(output == border, axis=2)))
        cubic = cv2.warpAffine(source, matrix, (11, 9), flags=cv2.INTER_CUBIC, borderValue=border)
        self.assertFalse(np.array_equal(output, cubic))

    def test_removed_outputs_and_positional_options_are_not_accepted(self):
        source = fixture()
        for name in ("return_aligned_image", "return_notch_visuals", "notch_visual_max_dimension", "notch_zoom_size_px", "notch_zoom_scale"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                angle(source, **{name: 1})
        with self.assertRaises(TypeError):
            angle(source, 90.0)


if __name__ == "__main__":
    unittest.main()
