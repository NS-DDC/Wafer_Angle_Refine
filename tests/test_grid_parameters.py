"""Grid-only option contracts and real-image-array resampling checks."""
from __future__ import annotations
import inspect
import unittest
from unittest import mock
import cv2
import numpy as np
from test_angle import MODULE, angle, synthetic_grid, independent_line_angles, FAST, SIZE

EXPECTED_DEFAULTS = {
    "roi_ratio": 0.55,
    "search_deg": 6.0,
    "coarse_step": 0.15,
    "fine_step": 0.02,
    "max_dim": 1100,
    "fft_max_dim": 1024,
    "agree_tol_deg": 0.50,
    "full_scan_deg": 44.0,
    "max_iter": 3,
    "min_angle_deg": 0.01,
    "alignment_interpolation": cv2.INTER_CUBIC,
    "alignment_border_value": (0, 0, 0),
}


class GridParameterTests(unittest.TestCase):
    def test_exact_public_names_defaults_and_keyword_only_contract(self):
        parameters = inspect.signature(angle).parameters
        actual = {name: value.default for name, value in parameters.items() if name != "src"}
        self.assertEqual(actual, EXPECTED_DEFAULTS)
        for name in EXPECTED_DEFAULTS:
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_nondefault_grid_analysis_options_still_align_both_axes(self):
        output = angle(
            synthetic_grid(-14.0), **FAST, roi_ratio=0.65, search_deg=5.0,
            coarse_step=0.20, fine_step=0.04, agree_tol_deg=0.75,
            full_scan_deg=44.0, min_angle_deg=0.03,
        )
        for residual in independent_line_angles(output):
            self.assertLess(abs(residual), 0.5)

    def test_invalid_options_return_original_copy_without_starting_analysis(self):
        cases = (
            ("roi_ratio", 0), ("roi_ratio", 1.01), ("roi_ratio", float("nan")),
            ("search_deg", 0), ("search_deg", 46), ("search_deg", float("inf")),
            ("coarse_step", 0), ("coarse_step", 7),
            ("fine_step", 0), ("fine_step", 0.2),
            ("max_dim", 0), ("max_dim", 63), ("max_dim", float("inf")),
            ("fft_max_dim", 0), ("fft_max_dim", 63),
            ("agree_tol_deg", 0), ("agree_tol_deg", 5.1),
            ("full_scan_deg", 0), ("full_scan_deg", 5), ("full_scan_deg", 46),
            ("max_iter", 0), ("max_iter", 11),
            ("min_angle_deg", -0.1), ("min_angle_deg", 1.1),
            ("alignment_interpolation", 5), ("alignment_interpolation", 0.5),
            ("alignment_border_value", (0, 0)),
            ("alignment_border_value", (0, 0, 256)),
            ("alignment_border_value", (0, 0, float("nan"))),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                source = np.arange(80 * 80 * 3, dtype=np.uint8).reshape(80, 80, 3)
                before = source.copy()
                with mock.patch.object(MODULE, "_measure_grid_angle") as measure:
                    output = angle(source, **{name: value})
                measure.assert_not_called()
                np.testing.assert_array_equal(output, before)
                np.testing.assert_array_equal(source, before)
                self.assertEqual(output.shape, source.shape)
                self.assertEqual(output.dtype, source.dtype)
                self.assertFalse(np.shares_memory(output, source))

    def test_nearest_interpolation_and_bgra_border_use_one_original_image_warp(self):
        bgr = synthetic_grid(12.0)
        source = np.dstack((bgr, np.full((SIZE, SIZE), 173, np.uint8)))
        before = source.copy()
        border = (12, 34, 56, 78)
        original_warp = cv2.warpAffine
        with mock.patch.object(MODULE.cv2, "warpAffine", wraps=original_warp) as resampling:
            output = angle(source, **FAST, alignment_interpolation=cv2.INTER_NEAREST, alignment_border_value=border)
        original_calls = [call for call in resampling.call_args_list if call.args and call.args[0] is source]
        self.assertEqual(len(original_calls), 1, "Resample the original image only once.")
        call = original_calls[0]
        self.assertEqual(call.kwargs["flags"], cv2.INTER_NEAREST)
        self.assertEqual(call.kwargs["borderMode"], cv2.BORDER_CONSTANT)
        self.assertEqual(call.kwargs["borderValue"], border)
        self.assertTrue(np.any(np.all(output == border, axis=2)))
        np.testing.assert_array_equal(source, before)
        self.assertEqual(output.shape, source.shape)
        self.assertEqual(output.dtype, source.dtype)
        self.assertFalse(np.shares_memory(source, output))

    def test_notch_output_and_positional_options_are_rejected(self):
        source = np.zeros((80, 80, 3), np.uint8)
        for name in ("notch_roi_center_px", "notch_reference_angle_deg", "return_aligned_image", "return_notch_visuals"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                angle(source, **{name: 1})
        with self.assertRaises(TypeError):
            angle(source, 0.55)

    def test_grid_outside_full_scan_returns_an_original_copy(self):
        source = synthetic_grid(25.0)
        before = source.copy()
        output = angle(source, **FAST, search_deg=6.0, full_scan_deg=6.0)
        np.testing.assert_array_equal(output, before)
        np.testing.assert_array_equal(source, before)
        self.assertFalse(np.shares_memory(output, source))

    def test_explicit_45_degree_scan_aligns_axes_without_requiring_unique_orientation(self):
        output = angle(synthetic_grid(45.0), **FAST, full_scan_deg=45.0)
        # At exactly 45 degrees, the two 90-degree-equivalent orientations
        # cannot be distinguished from the grid alone; do not test the marker.
        for residual in independent_line_angles(output):
            self.assertLess(abs(residual), 0.5)


if __name__ == "__main__":
    unittest.main()
