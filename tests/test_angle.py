"""Public angle(src) checks using known, locally generated wafer geometry.

These fixtures are synthetic. They do not certify production-camera images.
The alignment oracle uses an internal coloured landmark and the known input
rotation, rather than asking the production detector to validate itself.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ANGLE_MODULE = importlib.import_module("angle")
angle = ANGLE_MODULE.angle


def synthetic_wafer(*, notch: str = "triangle") -> np.ndarray:
    """Make a bottom notch and a separate red landmark at (600, 450)."""
    image = np.zeros((900, 900, 3), dtype=np.uint8)
    cv2.circle(image, (450, 450), 390, (110, 130, 145), -1, cv2.LINE_AA)
    cv2.circle(image, (450, 450), 390, (185, 195, 205), 5, cv2.LINE_AA)
    if notch == "triangle":
        polygon = np.asarray(((428, 864), (450, 825), (472, 864)), np.int32)
        cv2.fillPoly(image, [polygon], (0, 0, 0), cv2.LINE_AA)
    elif notch == "wide":
        cv2.ellipse(image, (450, 845), (70, 12), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
    elif notch != "none":
        raise ValueError(notch)
    cv2.rectangle(image, (590, 440), (610, 460), (0, 0, 255), -1)
    return image


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D((450.0, 450.0), degrees, 1.0)
    return cv2.warpAffine(
        image, matrix, (900, 900), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def landmark_angle(image: np.ndarray) -> float:
    """Measure a red marker whose unrotated direction is known to be right."""
    mask = (image[..., 2] > 225) & (image[..., 0] < 40) & (image[..., 1] < 40)
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise AssertionError("The independent red landmark disappeared.")
    return float(np.degrees(np.arctan2(ys.mean() - 450, xs.mean() - 450)))


class AngleTests(unittest.TestCase):
    def assert_image_contract(self, source, result):
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, source.dtype)
        self.assertFalse(np.shares_memory(result, source))
        self.assertTrue(result.flags.writeable)

    def test_signed_rotations_align_independent_landmark_and_known_warp(self):
        for applied in (-25.0, -12.0, 0.0, 12.0, 25.0):
            with self.subTest(applied=applied):
                source = rotate(synthetic_wafer(), applied)
                before = source.copy()
                result = angle(source)
                self.assert_image_contract(source, result)
                np.testing.assert_array_equal(source, before)
                self.assertLess(abs(landmark_angle(result)), 0.5)
                ideal = rotate(source, -applied)
                mean_error = np.mean(np.abs(result.astype(np.float32) - ideal.astype(np.float32)))
                self.assertLess(float(mean_error), 1.5)

    def test_wide_shallow_notch_alignment_uses_correct_rotation_sign(self):
        for applied in (-23.0, 23.0):
            with self.subTest(applied=applied):
                source = rotate(synthetic_wafer(notch="wide"), applied)
                result = angle(source)
                self.assertLess(abs(landmark_angle(result)), 0.5)

    def test_gray_bgr_bgra_keep_shape_dtype_and_input_pixels(self):
        base = synthetic_wafer()
        alpha = np.broadcast_to(np.linspace(20, 240, 900, dtype=np.uint8), (900, 900)).copy()
        formats = {
            "gray": cv2.cvtColor(base, cv2.COLOR_BGR2GRAY),
            "bgr": base,
            "bgra": np.dstack((base, alpha)),
        }
        for label, image in formats.items():
            with self.subTest(format=label):
                source = rotate(image, 17.0)
                before = source.copy()
                result = angle(source)
                self.assert_image_contract(source, result)
                np.testing.assert_array_equal(source, before)
                expected = rotate(source, -17.0)
                mean_error = np.mean(np.abs(result.astype(np.float32) - expected.astype(np.float32)))
                self.assertLess(float(mean_error), 1.5)
                if label == "bgra":
                    # This also rejects silently discarding/replacing alpha.
                    self.assertGreater(int(np.ptp(result[300:600, 300:600, 3])), 40)

    def test_readonly_noncontiguous_input_is_not_modified(self):
        source = rotate(synthetic_wafer(), -12.0)
        backing = np.zeros((900, 1800, 3), dtype=np.uint8)
        view = backing[:, ::2]
        view[:] = source
        view.setflags(write=False)
        self.assertFalse(view.flags.c_contiguous)
        result = angle(view)
        self.assert_image_contract(view, result)
        np.testing.assert_array_equal(view, source)
        self.assertLess(abs(landmark_angle(result)), 0.5)

    def test_unicode_file_path_matches_array_call(self):
        source = rotate(synthetic_wafer(), 12.0)
        with tempfile.TemporaryDirectory(prefix="angle-path-", dir=ROOT / "tests") as directory:
            path = Path(directory) / "웨이퍼 각도 테스트.png"
            success, encoded = cv2.imencode(".png", source)
            self.assertTrue(success)
            encoded.tofile(str(path))
            result = angle(str(path))
            self.assert_image_contract(source, result)
            np.testing.assert_array_equal(result, angle(source))

    def test_blank_images_raise_instead_of_claiming_alignment(self):
        for value in (0, 127, 255):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                angle(np.full((320, 400, 3), value, dtype=np.uint8))

    def test_circle_without_notch_is_rejected(self):
        with self.assertRaises(RuntimeError):
            angle(synthetic_wafer(notch="none"))

    def test_notch_outside_default_bottom_sector_is_rejected(self):
        with self.assertRaises(RuntimeError):
            angle(rotate(synthetic_wafer(), 60.0))

    def test_unsupported_dtypes_raise_type_error(self):
        for dtype in (np.float32, np.float64, np.uint16, np.int8, np.bool_):
            with self.subTest(dtype=np.dtype(dtype).name), self.assertRaises(TypeError):
                angle(np.zeros((40, 40, 3), dtype=dtype))

    def test_invalid_channel_or_rank_shapes_raise_value_error(self):
        for shape in ((40,), (40, 40, 1), (40, 40, 2), (40, 40, 5), (2, 40, 40, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                angle(np.zeros(shape, dtype=np.uint8))

    def test_empty_arrays_raise_value_error(self):
        for shape in ((0,), (0, 40), (40, 0, 3), (0, 40, 4)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                angle(np.empty(shape, dtype=np.uint8))

    def test_invalid_input_objects_raise_type_error(self):
        for source in (None, 123, [[0, 0], [0, 0]], object()):
            with self.subTest(type=type(source).__name__), self.assertRaises(TypeError):
                angle(source)

    def test_missing_and_undecodable_files_have_distinct_errors(self):
        with tempfile.TemporaryDirectory(prefix="angle-invalid-", dir=ROOT / "tests") as directory:
            folder = Path(directory)
            with self.assertRaises(FileNotFoundError):
                angle(str(folder / "없는 파일.png"))
            bad = folder / "깨진 이미지.png"
            bad.write_bytes(b"this is not image data")
            with self.assertRaises(ValueError):
                angle(str(bad))

    def test_public_export_has_one_positional_src_and_keyword_only_options(self):
        self.assertEqual(list(ANGLE_MODULE.__all__), ["angle"])
        parameters = list(inspect.signature(angle).parameters.values())
        self.assertEqual(parameters[0].name, "src")
        self.assertEqual(parameters[0].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual(parameters[0].default, inspect.Parameter.empty)
        self.assertEqual(len(parameters), 26)
        for parameter in parameters[1:]:
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertTrue(
                parameter.name.startswith("notch_")
                or parameter.name in ("alignment_interpolation", "alignment_border_value")
            )

    def test_copied_module_runs_in_isolated_python_without_upstream_or_output_files(self):
        with tempfile.TemporaryDirectory(prefix="angle-isolated-", dir=ROOT / "tests") as directory:
            folder = Path(directory)
            copied = folder / "angle.py"
            copied.write_bytes((ROOT / "angle.py").read_bytes())
            script = r'''
import importlib.util
from pathlib import Path
import sys
import cv2
import numpy as np
file = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("isolated_angle", file)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
source = np.zeros((900, 900, 3), np.uint8)
cv2.circle(source, (450, 450), 390, (110, 130, 145), -1, cv2.LINE_AA)
cv2.circle(source, (450, 450), 390, (185, 195, 205), 5, cv2.LINE_AA)
cv2.fillPoly(source, [np.array(((428, 864), (450, 825), (472, 864)), np.int32)], (0, 0, 0), cv2.LINE_AA)
before = sorted(str(p.relative_to(file.parent)) for p in file.parent.rglob("*"))
result = module.angle(source)
after = sorted(str(p.relative_to(file.parent)) for p in file.parent.rglob("*"))
assert result.shape == source.shape and result.dtype == source.dtype
assert not np.shares_memory(source, result)
assert before == after, (before, after)
assert module.__all__ == ["angle"]
print("isolated_angle_ok")
'''
            run = subprocess.run(
                [sys.executable, "-I", "-B", "-c", script, str(copied)],
                cwd=folder, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(run.stdout.strip(), "isolated_angle_ok")


if __name__ == "__main__":
    unittest.main()
