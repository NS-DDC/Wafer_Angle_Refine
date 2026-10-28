"""Independent synthetic internal-grid deskew tests, not camera validation."""
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
MODULE = importlib.import_module("angle")
angle = MODULE.angle
SIZE = 768
CENTER = SIZE / 2.0
FAST = {"max_dim": 600, "fft_max_dim": 512, "max_iter": 2}
MARKER_ANGLE = float(np.degrees(np.arctan2(24.0, 150.0)))


def synthetic_grid(rotation_deg=0.0, *, dark=False, axes="both", notch=False, marker=True, isoluminant=False):
    """Draw rotated streets analytically; no implementation or image warp used."""
    ys, xs = np.indices((SIZE, SIZE), dtype=np.float32)
    dx, dy = xs - CENTER, ys - CENTER
    theta = np.radians(rotation_deg)
    u = dx * np.cos(theta) - dy * np.sin(theta)
    v = dx * np.sin(theta) + dy * np.cos(theta)
    vertical = np.abs((u + 32.0) % 64.0 - 32.0)
    horizontal = np.abs((v + 36.0) % 72.0 - 36.0)
    if axes == "both":
        distance = np.minimum(vertical, horizontal)
    elif axes == "vertical":
        distance = vertical
    elif axes == "horizontal":
        distance = horizontal
    elif axes == "none":
        distance = np.full_like(vertical, 100.0)
    else:
        raise ValueError(axes)
    if isoluminant:
        # These two exact uint8 colours both convert to gray 29 in OpenCV.
        coverage = (distance <= 2.5).astype(np.float32)[..., None]
        background = np.asarray((0, 49, 0), np.float32)
        street = np.asarray((255, 0, 0), np.float32)
    else:
        coverage = np.clip(3.0 - distance, 0.0, 1.0)[..., None]
        background = np.asarray((242, 242, 242) if dark else (30, 150, 45), np.float32)
        street = np.asarray((15, 15, 15) if dark else (255, 255, 255), np.float32)
    image = np.rint(background + coverage * (street - background)).astype(np.uint8)
    image[dx * dx + dy * dy > 338.0**2] = (5, 8, 10)
    cv2.circle(image, (384, 384), 338, (205, 215, 220), 8, cv2.LINE_AA)
    if notch:
        # This notch stays at the bottom independently of the grid rotation.
        cv2.fillPoly(image, [np.asarray(((362, 750), (384, 704), (406, 750)), np.int32)], (5, 8, 10), cv2.LINE_AA)
    if marker:
        point = (int(round(CENTER + 150 * np.cos(theta) + 24 * np.sin(theta))),
                 int(round(CENTER - 150 * np.sin(theta) + 24 * np.cos(theta))))
        cv2.circle(image, point, 7, (0, 0, 255), -1, cv2.LINE_AA)
    return image


def independent_line_angles(image, *, dark=False, blue=False):
    """Fit only the known central streets, independently of detector helpers."""
    if blue:
        street = (image[..., 0] > 180) & (image[..., 1] < 15)
    elif image.ndim == 2:
        street = image < 60 if dark else image > 220
    else:
        rgb = image[..., :3]
        street = np.all(rgb < 60, axis=2) if dark else np.all(rgb > 220, axis=2)
    vertical, horizontal = [], []
    for coordinate in range(244, 525):
        offset = coordinate - 384
        if abs((offset + 36) % 72 - 36) > 10:
            points = np.flatnonzero(street[coordinate, 374:395]) + 374
            if 2 <= len(points) <= 14:
                vertical.append((coordinate, points.mean()))
        if abs((offset + 32) % 64 - 32) > 10:
            points = np.flatnonzero(street[374:395, coordinate]) + 374
            if 2 <= len(points) <= 14:
                horizontal.append((coordinate, points.mean()))
    if min(len(vertical), len(horizontal)) < 100:
        raise AssertionError("Both central grid directions were not aligned or visible.")
    angles = []
    for coordinates in (vertical, horizontal):
        independent, dependent = np.asarray(coordinates).T
        angles.append(float(np.degrees(np.arctan(np.polyfit(independent, dependent, 1)[0]))))
    return angles


def marker_residual(image):
    mask = (image[..., 2] > 225) & (image[..., 0] < 40) & (image[..., 1] < 40)
    ys, xs = np.nonzero(mask)
    if len(xs) < 80:
        raise AssertionError("Independent red landmark disappeared.")
    observed = float(np.degrees(np.arctan2(ys.mean() - CENTER, xs.mean() - CENTER)))
    return (observed - MARKER_ANGLE + 180.0) % 360.0 - 180.0


class AngleTests(unittest.TestCase):
    def assert_image_contract(self, source, result):
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, source.dtype)
        self.assertFalse(np.shares_memory(result, source))
        self.assertTrue(result.flags.writeable)

    def assert_original_copy(self, source, before, result):
        self.assert_image_contract(source, result)
        np.testing.assert_array_equal(source, before)
        np.testing.assert_array_equal(result, before)

    def assert_grid_aligned(self, result, *, dark=False):
        for residual in independent_line_angles(result, dark=dark):
            self.assertLess(abs(residual), 0.5)
        self.assertLess(abs(marker_residual(result)), 0.5)

    def test_signed_rotations_align_both_axes_and_independent_landmark(self):
        for applied in (-25.0, -12.0, 0.0, 12.0, 25.0):
            with self.subTest(applied=applied):
                source = synthetic_grid(applied)
                before = source.copy()
                result = angle(source, **FAST)
                self.assert_image_contract(source, result)
                np.testing.assert_array_equal(source, before)
                self.assert_grid_aligned(result)

    def test_default_call_aligns_dark_grid_on_light_dies(self):
        source = synthetic_grid(17.0, dark=True)
        result = angle(source)
        self.assert_image_contract(source, result)
        self.assert_grid_aligned(result, dark=True)

    def test_fixed_bottom_notch_and_circular_rim_do_not_override_grid(self):
        result = angle(synthetic_grid(12.0, notch=True), **FAST)
        self.assert_grid_aligned(result)

    def test_noisy_grid_keeps_both_axis_alignment(self):
        source = synthetic_grid(-7.0)
        rng = np.random.default_rng(20261027)
        source = np.clip(source.astype(np.float32) + rng.normal(0.0, 8.0, source.shape), 0, 255).astype(np.uint8)
        self.assert_grid_aligned(angle(source, **FAST))

    def test_isoluminant_colour_grid_aligns_when_gray_has_no_grid_contrast(self):
        source = synthetic_grid(9.0, isoluminant=True)
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(np.unique(gray[250:350, 250:500]), [29])
        output = angle(source, **FAST)
        for residual in independent_line_angles(output, blue=True):
            self.assertLess(abs(residual), 0.5)
        self.assertLess(abs(marker_residual(output)), 0.5)
        before = gray.copy()
        self.assert_original_copy(gray, before, angle(gray, **FAST))

    def test_gray_bgr_bgra_preserve_shape_dtype_alpha_and_input(self):
        base = synthetic_grid(11.0)
        alpha = np.broadcast_to(np.linspace(20, 240, SIZE, dtype=np.uint8), (SIZE, SIZE)).copy()
        inputs = (cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), base, np.dstack((base, alpha)))
        for source in inputs:
            with self.subTest(shape=source.shape):
                before = source.copy()
                result = angle(source, **FAST)
                self.assert_image_contract(source, result)
                np.testing.assert_array_equal(source, before)
                for residual in independent_line_angles(result):
                    self.assertLess(abs(residual), 0.5)
                if source.ndim == 3:
                    self.assert_grid_aligned(result)
                    if source.shape[2] == 4:
                        expected_alpha = cv2.warpAffine(source[..., 3], cv2.getRotationMatrix2D((CENTER, CENTER), -11.0, 1.0), (SIZE, SIZE), flags=cv2.INTER_CUBIC)
                        error = np.mean(np.abs(expected_alpha.astype(np.float32) - result[..., 3].astype(np.float32)))
                        self.assertLess(float(error), 1.0)

    def test_readonly_noncontiguous_input_is_not_modified(self):
        source = synthetic_grid(-8.0)
        backing = np.zeros((SIZE, SIZE * 2, 3), np.uint8)
        view = backing[:, ::2]
        view[:] = source
        view.setflags(write=False)
        self.assertFalse(view.flags.c_contiguous)
        result = angle(view, **FAST)
        self.assert_image_contract(view, result)
        np.testing.assert_array_equal(view, source)
        self.assert_grid_aligned(result)

    def test_unicode_file_path_matches_array_call(self):
        source = synthetic_grid(8.0)
        with tempfile.TemporaryDirectory(prefix="angle-path-", dir=ROOT / "tests") as directory:
            path = Path(directory) / "웨이퍼 격자 테스트.png"
            success, encoded = cv2.imencode(".png", source)
            self.assertTrue(success)
            encoded.tofile(str(path))
            result = angle(str(path), **FAST)
            self.assert_image_contract(source, result)
            np.testing.assert_array_equal(result, angle(source, **FAST))

    def test_blank_images_and_circle_without_internal_grid_return_original_copies(self):
        cases = (np.zeros((256, 320, 3), np.uint8), np.full((256, 320, 3), 255, np.uint8), synthetic_grid(axes="none", marker=False))
        for index, source in enumerate(cases):
            with self.subTest(case=index):
                before = source.copy()
                self.assert_original_copy(source, before, angle(source, **FAST))

    def test_single_direction_stripes_return_original_copies(self):
        for axes in ("vertical", "horizontal"):
            with self.subTest(axes=axes):
                source = synthetic_grid(12.0, axes=axes, marker=False)
                before = source.copy()
                self.assert_original_copy(source, before, angle(source, **FAST))

    def test_noise_and_equal_competing_grids_return_original_copies(self):
        noise = np.random.default_rng(3512).integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
        conflicting = np.maximum(synthetic_grid(-12, marker=False), synthetic_grid(12, marker=False))
        for label, source in (("noise", noise), ("competing", conflicting)):
            with self.subTest(case=label):
                before = source.copy()
                self.assert_original_copy(source, before, angle(source, **FAST))

    def test_unsupported_dtypes_raise_type_error(self):
        for dtype in (np.float32, np.float64, np.uint16, np.int8, np.bool_):
            with self.subTest(dtype=np.dtype(dtype).name), self.assertRaises(TypeError):
                angle(np.zeros((40, 40, 3), dtype=dtype))

    def test_invalid_or_empty_shapes_raise_value_error(self):
        for shape in ((40,), (40, 40, 1), (40, 40, 2), (40, 40, 5), (2, 40, 40, 3), (0,), (0, 40), (40, 0, 3), (0, 40, 4)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                angle(np.zeros(shape, dtype=np.uint8))

    def test_invalid_objects_missing_files_and_bad_decodes_have_specific_errors(self):
        for source in (None, 123, [[0, 0], [0, 0]], object()):
            with self.subTest(type=type(source).__name__), self.assertRaises(TypeError):
                angle(source)
        with tempfile.TemporaryDirectory(prefix="angle-invalid-", dir=ROOT / "tests") as directory:
            folder = Path(directory)
            with self.assertRaises(FileNotFoundError):
                angle(str(folder / "없는 파일.png"))
            bad = folder / "깨진 이미지.png"
            bad.write_bytes(b"this is not image data")
            with self.assertRaises(ValueError):
                angle(str(bad))

    def test_public_export_has_one_positional_src_and_keyword_only_options(self):
        self.assertEqual(list(MODULE.__all__), ["angle"])
        parameters = list(inspect.signature(angle).parameters.values())
        self.assertEqual(parameters[0].name, "src")
        self.assertEqual(parameters[0].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual(parameters[0].default, inspect.Parameter.empty)
        self.assertEqual(len(parameters), 13)
        for parameter in parameters[1:]:
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_copied_module_runs_without_upstream_files_or_writing_outputs(self):
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
before = sorted(str(p.relative_to(file.parent)) for p in file.parent.rglob("*"))
spec = importlib.util.spec_from_file_location("isolated_angle", file)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
source = np.full((640, 640, 3), (30, 150, 45), np.uint8)
for x in range(32, 640, 64):
    cv2.line(source, (x, 0), (x, 639), (255, 255, 255), 5)
for y in range(32, 640, 72):
    cv2.line(source, (0, y), (639, y), (255, 255, 255), 5)
result = module.angle(source, max_dim=600, fft_max_dim=512, max_iter=2)
after = sorted(str(p.relative_to(file.parent)) for p in file.parent.rglob("*"))
assert result.shape == source.shape and result.dtype == source.dtype
assert not np.shares_memory(source, result)
assert before == after, (before, after)
assert module.__all__ == ["angle"]
print("isolated_angle_ok")
'''
            run = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(copied)], cwd=folder, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(run.stdout.strip(), "isolated_angle_ok")


if __name__ == "__main__":
    unittest.main()
