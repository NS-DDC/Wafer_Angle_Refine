# Image-grid core: NS-DDC/Wafer_color_v6, commit 9f3b7bf,
# wafer_color_v6_claude.py (Lab edge energy, projection and FFT helpers).
# Search settings also follow NS-DDC/Wafer_Via, commit 5adf92e,
# codex/wafer_via_die_render.py, sector 85. No external project import is used.
"""Deskew the internal orthogonal grid: aligned = angle(src).

The source is resampled once on success. If correction fails after loading a
valid image, an unchanged copy is returned so the next pipeline step can run.
The nearest axis is used, with 90-degree ambiguity.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

__all__ = ["angle"]
_EPS = 1e-12


def _load_image(src) -> np.ndarray:
    if isinstance(src, np.ndarray):
        image = src
    elif isinstance(src, (str, Path)):
        path = Path(src)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        encoded = np.fromfile(path, dtype=np.uint8)
        if encoded.size == 0:
            raise ValueError(f"Image file is empty: {path}")
        try:
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        except cv2.error as exc:
            raise ValueError(f"Image file could not be decoded: {path}") from exc
        if image is None:
            raise ValueError(f"Image file could not be decoded: {path}")
    else:
        raise TypeError("src must be a uint8 ndarray, str, or pathlib.Path.")

    if image.dtype != np.uint8:
        raise TypeError("Image dtype must be uint8.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("Image shape must be (H,W), (H,W,3) BGR, or (H,W,4) BGRA.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image must have nonzero height and width.")

    return image


def _bgr_to_lab_f32(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> Lab float32 (OpenCV 8U Lab 스케일: L 0..255, a/b 0..255@128)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _color_edge_energy(roi_bgr: np.ndarray) -> np.ndarray:
    """Lab 3채널 그래디언트 에너지 맵 (float32).

    V5 는 회색조 |Sobel_x| 만 봤기 때문에, 밝기는 같고 색만 다른 street
    (예: 회색 die 옆의 같은 밝기 갈색 street)를 아예 못 본다.
    V6 는 L/a/b 모두의 그래디언트를 합쳐 **어떤 색 대비든** 에너지로 잡는다.
    """
    lab = _bgr_to_lab_f32(roi_bgr)
    acc = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        ch = lab[:, :, c]
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        acc += gx * gx + gy * gy
    return np.sqrt(acc)


def _angle_projection_score(E: np.ndarray, angle_deg: float,
                            inner: int) -> float:
    """에너지 맵을 angle 만큼 회전 후, 행/열 프로파일의 '뾰족함'.

    격자가 축과 나란해지면 열/행 합 프로파일이 뾰족해진다(분산 최대).
    회전으로 생기는 검은 모서리는 항상 유효한 내접 정사각(inner)만 잘라 배제.
    평균 제곱으로 정규화해 회전에 따른 밝기 변화가 점수를 왜곡하지 않게 한다.
    """
    h, w = E.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    rot = cv2.warpAffine(E, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    oy = (h - inner) // 2
    ox = (w - inner) // 2
    sub = rot[oy:oy + inner, ox:ox + inner]
    col = sub.mean(axis=0)
    row = sub.mean(axis=1)
    sc = 0.0
    for pr in (col, row):
        m = float(pr.mean())
        if m > _EPS:
            sc += float(pr.var()) / (m * m)
    return sc


def _angle_from_fft(E: np.ndarray) -> Optional[float]:
    """2D FFT 스펙트럼의 90도-주기 방향 원형평균 -> 격자 기울기 (deg).

    projection 탐색과 **독립적인** 두 번째 단서. 두 값이 일치하면 신뢰도 상승.
    격자는 0deg/90deg 두 방향에 에너지를 만들므로 alpha 를 4배각으로 접어
    (mod 90) 가중 원형평균을 취한다.
    """
    h, w = E.shape[:2]
    if min(h, w) < 64:
        return None
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.fftshift(np.abs(np.fft.fft2((E - E.mean()) * win)))
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    u = (xx - cx).astype(np.float64)
    v = (yy - cy).astype(np.float64)
    rad = np.sqrt(u * u + v * v)
    rmax = min(cx, cy)
    band = (rad > rmax * 0.03) & (rad < rmax * 0.90)
    if not band.any():
        return None
    wgt = F[band].astype(np.float64) ** 2
    alpha = np.arctan2(v[band], u[band])
    acc = complex(np.sum(wgt * np.exp(4j * alpha)))
    if abs(acc) < _EPS:
        return None
    ang = math.degrees(np.angle(acc)) / 4.0        # -45 .. +45
    return float(((ang + 45.0) % 90.0) - 45.0)




def _fft_grid_angle(energy: np.ndarray) -> Optional[float]:
    """V7 radial-peak FFT cue; use it only with strong fourfold concentration.

    A narrow dominant-frequency ring separates repeated grid spacing from
    unrelated die texture. If that ring is not directional, keep the V6
    whole-spectrum estimate rather than forcing a radial-peak angle.
    """
    size = min(energy.shape)
    window = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    square = energy[:size, :size]
    spectrum = np.fft.fftshift(np.fft.fft2((square - float(square.mean())) * window))
    power = spectrum.real ** 2 + spectrum.imag ** 2
    yy, xx = np.mgrid[:size, :size]
    dx, dy = xx - size // 2, yy - size // 2
    radius = np.hypot(dx, dy)
    minimum, maximum = max(4.0, size * 0.012), size * 0.45
    band = (radius >= minimum) & (radius <= maximum)
    radial = np.bincount(radius.astype(np.int32)[band], weights=power[band])
    if len(radial) and float(radial.max()) > 1e-9:
        peak = max(float(np.argmax(radial)), minimum + 1.0)
        annulus = (np.abs(radius - peak) <= max(2.0, peak * 0.20)) & band
        weights = power[annulus]
        mass = float(weights.sum())
        if mass > 1e-9:
            phase = np.arctan2(dy[annulus], dx[annulus])
            vector = np.sum(weights * np.exp(4j * phase))
            if abs(vector) / mass >= 0.85:
                return _fold_angle(float(np.degrees(np.angle(vector)) / 4.0))
    return _angle_from_fft(energy)


def _fold_angle(value: float) -> float:
    return float((float(value) + 45.0) % 90.0 - 45.0)


def _angle_distance(first: float, second: float) -> float:
    return abs(_fold_angle(first - second))


def _profile_is_periodic(profile: np.ndarray) -> bool:
    """Require several separated, coherent lines along one image axis."""
    values = cv2.GaussianBlur(np.asarray(profile, np.float32).reshape(1, -1), (0, 0), 0.8).ravel()
    baseline = float(np.median(values))
    high = float(np.percentile(values, 99))
    if high - baseline < 3.0:
        return False
    average = float(values.mean())
    modulation = float(values.var()) / max(average * average, 1e-12)
    if average <= 1e-6 or modulation < 0.008:
        return False
    active = np.flatnonzero(values >= baseline + (high - baseline) * 0.35)
    groups = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
    centres = [float(group.mean()) for group in groups if len(group)]
    if len(centres) < 3 or centres[-1] - centres[0] < len(values) * 0.35:
        return False
    # Fourier fourfold symmetry alone also accepts one family of stripes.
    signal = values.astype(np.float64) - float(values.mean())
    correlation = np.correlate(signal, signal, mode="full")[len(signal) - 1:]
    if correlation[0] <= 1e-9:
        return False
    correlation /= correlation[0]
    start, stop = 4, len(values) // 2
    if stop <= start + 2:
        return False
    peaks = np.flatnonzero((correlation[start:stop] >= correlation[start - 1:stop - 1])
                          & (correlation[start:stop] >= correlation[start + 1:stop + 1])) + start
    # Textured dies raise the profile baseline without erasing repeated street
    # positions. Permit weaker modulation only with much stronger periodicity.
    required = 0.60 if modulation < 0.08 or high - baseline < baseline * 0.40 else 0.25
    return bool(len(peaks) and float(np.max(correlation[peaks])) >= required)


def _aligned_profiles(energy: np.ndarray, correction: float, inner: int):
    height, width = energy.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), correction, 1.0)
    rotated = cv2.warpAffine(energy, matrix, (width, height), flags=cv2.INTER_LINEAR)
    top, left = (height - inner) // 2, (width - inner) // 2
    valid = rotated[top:top + inner, left:left + inner]
    return valid.mean(axis=0), valid.mean(axis=1)


def _search_projection(energy: np.ndarray, *, centre: float, span: float,
                       coarse_step: float, fine_step: float, inner: int, limit: float):
    """V6 coarse/fine/parabolic search, retaining a fixed valid crop."""
    low, high = max(-limit, centre - span), min(limit, centre + span)
    if low > high:
        raise RuntimeError("The Fourier orientation lies outside full_scan_deg.")
    coarse = np.arange(low, high + 1e-9, coarse_step)
    if coarse[-1] < high - 1e-9:
        coarse = np.append(coarse, high)
    scores = np.asarray([_angle_projection_score(energy, float(value), inner) for value in coarse])
    best = int(np.argmax(scores))
    fine_low, fine_high = max(-limit, coarse[best] - coarse_step), min(limit, coarse[best] + coarse_step)
    fine = np.arange(fine_low, fine_high + 1e-9, fine_step)
    if fine[-1] < fine_high - 1e-9:
        fine = np.append(fine, fine_high)
    fine_scores = np.asarray([_angle_projection_score(energy, float(value), inner) for value in fine])
    index = int(np.argmax(fine_scores));result = float(fine[index])
    if 0 < index < len(fine_scores) - 1:
        denominator = fine_scores[index - 1] - 2.0 * fine_scores[index] + fine_scores[index + 1]
        if abs(denominator) > _EPS:
            offset = 0.5 * (fine_scores[index - 1] - fine_scores[index + 1]) / denominator
            if -1.0 < offset < 1.0:
                result += float(offset * fine_step)
    score = float(fine_scores[index])
    ambiguous = any(_angle_distance(float(value), result) >= 2.0 and strength >= score * 0.90
                    for value, strength in zip(coarse, scores))
    return _fold_angle(float(np.clip(result, -limit, limit))), score, ambiguous


def _measure_grid_angle(image: np.ndarray, *, roi_ratio: float, search_deg: float,
                        coarse_step: float, fine_step: float, max_dim: int,
                        fft_max_dim: int, agree_tol_deg: float, full_scan_deg: float) -> float:
    height, width = image.shape[:2]
    size = min(min(height, width), int(round(min(height, width) * roi_ratio)))
    if size < 64:
        raise RuntimeError("The central grid ROI is too small; increase roi_ratio or use a larger image.")
    left, top = (width - size) // 2, (height - size) // 2
    roi = image[top:top + size, left:left + size]
    if size > max_dim:
        roi = cv2.resize(roi, (max_dim, max_dim), interpolation=cv2.INTER_AREA)
    lab = _bgr_to_lab_f32(roi)
    contrast = np.max(lab, axis=(0, 1)) - np.min(lab, axis=(0, 1))
    if float(np.max(contrast)) < 6.0:
        raise RuntimeError("The image has insufficient colour contrast for a grid angle.")
    energy = _color_edge_energy(roi)
    if (not np.all(np.isfinite(energy)) or float(np.max(energy)) < 3.0
            or int(np.count_nonzero(energy >= 3.0)) < max(16, int(energy.size * 0.001))):
        raise RuntimeError("The central ROI has no usable internal grid edges.")
    fft_energy = energy
    if max(energy.shape) > fft_max_dim:
        fft_energy = cv2.resize(energy, (fft_max_dim, fft_max_dim), interpolation=cv2.INTER_AREA)
    fft_angle = _fft_grid_angle(fft_energy)
    if fft_angle is None:
        raise RuntimeError("The internal grid has no reliable Fourier orientation.")
    if abs(fft_angle) > full_scan_deg + agree_tol_deg:
        raise RuntimeError("The internal grid angle is outside full_scan_deg.")
    # Inscribed crop is valid for all orientations, excluding black padding.
    inner = int(min(energy.shape) / math.sqrt(2.0)) - 2
    if inner < 32:
        raise RuntimeError("Too few valid interior pixels for grid-angle estimation.")
    candidate = _search_projection(energy, centre=0.0, span=search_deg,
                                   coarse_step=coarse_step, fine_step=fine_step, inner=inner, limit=full_scan_deg)
    if _angle_distance(candidate[0], fft_angle) > agree_tol_deg:
        broad = _search_projection(energy, centre=0.0, span=full_scan_deg,
                                   coarse_step=max(coarse_step * 2.0, 0.30),
                                   fine_step=fine_step, inner=inner, limit=full_scan_deg)
        # The independent FFT-local search obeys the same explicit scan limit.
        fft_candidate = _search_projection(energy, centre=fft_angle, span=max(1.0, coarse_step * 3.0),
                                           coarse_step=coarse_step, fine_step=fine_step, inner=inner, limit=full_scan_deg)
        candidate = max((candidate, broad, fft_candidate), key=lambda item: item[1])
        if broad[2] and broad[1] >= candidate[1] * 0.90:
            raise RuntimeError("Several incompatible internal grid angles have comparable support.")
    correction, _, ambiguous = candidate
    if ambiguous or _angle_distance(correction, fft_angle) > agree_tol_deg:
        raise RuntimeError("Projection and Fourier grid-angle evidence is ambiguous or disagrees.")
    columns, rows = _aligned_profiles(energy, correction, inner)
    if not _profile_is_periodic(columns) or not _profile_is_periodic(rows):
        raise RuntimeError("Both horizontal and vertical repeating grid lines are required.")
    # Each axis must independently prefer the same correction. A summed
    # projection and a fourfold FFT can otherwise hide a non-orthogonal grid.
    offsets = np.linspace(-2.0, 2.0, 17)
    local_angles = np.clip(correction + offsets, -full_scan_deg, full_scan_deg)
    axis_scores = []
    for candidate_angle in local_angles:
        axis_profiles = _aligned_profiles(energy, float(candidate_angle), inner)
        axis_scores.append([float(profile.var()) / max(float(profile.mean()) ** 2, 1e-12)
                            for profile in axis_profiles])
    axis_scores = np.asarray(axis_scores)
    preferred = []
    for axis in range(2):
        index = int(np.argmax(axis_scores[:, axis]))
        value = float(local_angles[index])
        if 0 < index < len(local_angles) - 1:
            before, current, after = axis_scores[index - 1:index + 2, axis]
            denominator = before - 2.0 * current + after
            if abs(denominator) > 1e-12:
                offset = 0.5 * (before - after) / denominator
                if abs(offset) < 1.0:
                    value += float(offset * 0.25)
        preferred.append(value)
    if (_angle_distance(preferred[0], preferred[1]) > agree_tol_deg
            or any(_angle_distance(value, correction) > agree_tol_deg for value in preferred)):
        raise RuntimeError("The two grid axes do not support one orthogonal correction.")
    return correction


def _finite_parameter(name: str, value, low: float, high: float, *, inclusive_low: bool = True) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number) or number > high or (number < low if inclusive_low else number <= low):
        raise ValueError(f"{name} is outside its supported range.")
    return number


def angle(
    src,
    *,
    roi_ratio: float = 0.55,
    search_deg: float = 6.0,
    coarse_step: float = 0.15,
    fine_step: float = 0.02,
    max_dim: int = 1100,
    fft_max_dim: int = 1024,
    agree_tol_deg: float = 0.50,
    full_scan_deg: float = 44.0,
    max_iter: int = 3,
    min_angle_deg: float = 0.01,
    alignment_interpolation: int = cv2.INTER_CUBIC,
    alignment_border_value = (0, 0, 0),
) -> np.ndarray:
    """Return the image with its internal grid made horizontal and vertical.

    Input: uint8 grayscale/BGR/BGRA ndarray or Unicode str/Path filename.
    Output: a new ndarray with identical shape/dtype, rotated about image centre.
    roi_ratio is the central analysis square side / shortest image side.
    Angles use degrees; dimensions use analysis pixels. The nearest axis is
    selected modulo 90 degrees, without assigning an absolute die direction.
    Both axes must repeat visibly. After loading a valid image, any ordinary
    exception during parameter validation or correction returns image.copy().
    This includes insufficient/conflicting grid evidence and OpenCV errors.
    Unreadable paths and unsupported input arrays still raise an input error:
    no valid original image is available in those cases. Process interrupts
    such as KeyboardInterrupt and SystemExit are not caught.
    """
    image = _load_image(src)
    try:
        roi_ratio = _finite_parameter("roi_ratio", roi_ratio, 0.0, 1.0, inclusive_low=False)
        search_deg = _finite_parameter("search_deg", search_deg, 0.0, 45.0, inclusive_low=False)
        full_scan_deg = _finite_parameter("full_scan_deg", full_scan_deg, search_deg, 45.0)
        coarse_step = _finite_parameter("coarse_step", coarse_step, 0.01, min(5.0, search_deg))
        fine_step = _finite_parameter("fine_step", fine_step, 0.001, coarse_step)
        agree_tol_deg = _finite_parameter("agree_tol_deg", agree_tol_deg, 0.0, 5.0, inclusive_low=False)
        min_angle_deg = _finite_parameter("min_angle_deg", min_angle_deg, 0.0, 1.0)
        integers = []
        for name, value, low, high in (("max_dim", max_dim, 64, 8192),
                                       ("fft_max_dim", fft_max_dim, 64, 8192),
                                       ("max_iter", max_iter, 1, 10)):
            number = _finite_parameter(name, value, low, high)
            if number != int(number):
                raise ValueError(f"{name} must be an integer.")
            integers.append(int(number))
        max_dim, fft_max_dim, max_iter = integers
        interpolation = _finite_parameter("alignment_interpolation", alignment_interpolation, 0, 4)
        if interpolation != int(interpolation):
            raise ValueError("alignment_interpolation must be an OpenCV interpolation constant.")
        alignment_interpolation = int(interpolation)
        try:
            border = tuple(float(value) for value in alignment_border_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("alignment_border_value must contain 1, 3, or 4 channel values.") from exc
        if len(border) not in (1, 3, 4) or any(not math.isfinite(value) or not 0 <= value <= 255 for value in border):
            raise ValueError("alignment_border_value must contain 1, 3, or 4 finite values from 0 to 255.")
        if image.ndim == 2:
            analysis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            analysis = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            analysis = image
        height, width = image.shape[:2]
        center = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
        total = 0.0
        for _ in range(max_iter):
            if abs(total) > 1e-12:
                matrix = cv2.getRotationMatrix2D(center, total, 1.0)
                view = cv2.warpAffine(analysis, matrix, (width, height), flags=cv2.INTER_LINEAR)
            else:
                view = analysis
            delta = _measure_grid_angle(view, roi_ratio=roi_ratio, search_deg=search_deg,
                                         coarse_step=coarse_step, fine_step=fine_step, max_dim=max_dim,
                                         fft_max_dim=fft_max_dim, agree_tol_deg=agree_tol_deg,
                                         full_scan_deg=full_scan_deg)
            updated = _fold_angle(total + delta)
            if abs(updated) > full_scan_deg + 1e-9:
                raise RuntimeError("The accumulated grid correction exceeds full_scan_deg.")
            total = updated
            if abs(delta) <= min_angle_deg:
                break
        matrix = cv2.getRotationMatrix2D(center, total, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), flags=alignment_interpolation,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    except Exception:
        # Pipeline continuity is intentional. Discard any partial correction
        # and keep downstream writes from modifying the caller's source array.
        return image.copy()
