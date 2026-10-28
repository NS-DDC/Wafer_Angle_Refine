"""Processing errors preserve the loaded original for the next pipeline step."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from test_angle import MODULE, ROOT, angle, synthetic_grid


class ErrorFallbackTests(unittest.TestCase):
    def assert_original_copy(self, source, before, output):
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, source.shape)
        self.assertEqual(output.dtype, source.dtype)
        self.assertFalse(np.shares_memory(output, source))
        self.assertTrue(output.flags.writeable)
        np.testing.assert_array_equal(source, before)
        np.testing.assert_array_equal(output, before)

    def test_failure_after_an_iteration_returns_original_not_partial_rotation(self):
        source = synthetic_grid(12.0)
        before = source.copy()
        with mock.patch.object(
            MODULE, "_measure_grid_angle", side_effect=[7.0, RuntimeError("second iteration failed")],
        ) as measure:
            output = angle(source, max_iter=3)
        self.assertEqual(measure.call_count, 2)
        second_view = measure.call_args_list[1].args[0]
        self.assertFalse(np.array_equal(second_view, before), "The failure must occur after an actual analysis rotation.")
        self.assert_original_copy(source, before, output)

    def test_opencv_conversion_error_returns_original_gray_or_bgra(self):
        gray = np.arange(48 * 64, dtype=np.uint8).reshape(48, 64)
        bgra = np.dstack((gray, gray // 2, 255 - gray, gray))
        for source in (gray, bgra):
            with self.subTest(shape=source.shape):
                before = source.copy()
                with mock.patch.object(MODULE.cv2, "cvtColor", side_effect=cv2.error("conversion failed")), \
                        mock.patch.object(MODULE, "_measure_grid_angle") as measure:
                    output = angle(source)
                measure.assert_not_called()
                self.assert_original_copy(source, before, output)

    def test_opencv_final_warp_error_returns_original(self):
        source = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        before = source.copy()
        with mock.patch.object(MODULE, "_measure_grid_angle", return_value=0.0), \
                mock.patch.object(MODULE.cv2, "warpAffine", side_effect=cv2.error("final warp failed")) as warp:
            output = angle(source)
        warp.assert_called_once()
        self.assertIs(warp.call_args.args[0], source)
        self.assert_original_copy(source, before, output)

    def test_generic_processing_failure_preserves_formats_and_readonly_view(self):
        bgr = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bgra = np.dstack((bgr, gray))
        backing = np.zeros((48, 128, 3), np.uint8)
        readonly_view = backing[:, ::2]
        readonly_view[:] = bgr
        readonly_view.setflags(write=False)
        self.assertFalse(readonly_view.flags.c_contiguous)
        for label, source in (("gray", gray), ("bgr", bgr), ("bgra", bgra), ("readonly", readonly_view)):
            with self.subTest(format=label):
                before = source.copy()
                with mock.patch.object(MODULE, "_measure_grid_angle", side_effect=LookupError("processing failed")):
                    output = angle(source)
                self.assert_original_copy(source, before, output)

    def test_system_exit_and_keyboard_interrupt_are_not_swallowed(self):
        source = np.zeros((80, 80, 3), np.uint8)
        for exception in (SystemExit, KeyboardInterrupt):
            with self.subTest(exception=exception.__name__), \
                    mock.patch.object(MODULE, "_measure_grid_angle", side_effect=exception("stop")), \
                    self.assertRaises(exception):
                angle(source)

    def test_path_fallback_returns_loaded_pixels_in_a_separate_array(self):
        source = np.arange(48 * 64 * 4, dtype=np.uint8).reshape(48, 64, 4)
        with tempfile.TemporaryDirectory(prefix="angle-fallback-", dir=ROOT / "tests") as directory:
            path = Path(directory) / "원본 이미지.png"
            success, encoded = cv2.imencode(".png", source)
            self.assertTrue(success)
            encoded.tofile(str(path))
            loaded = []
            load_image = MODULE._load_image

            def remember_loaded_image(value):
                image = load_image(value)
                loaded.append(image)
                return image

            with mock.patch.object(MODULE, "_load_image", side_effect=remember_loaded_image), \
                    mock.patch.object(MODULE, "_measure_grid_angle", side_effect=RuntimeError("no reliable grid")):
                output = angle(str(path))
            self.assertEqual(len(loaded), 1)
            self.assert_original_copy(loaded[0], source, output)


if __name__ == "__main__":
    unittest.main()
