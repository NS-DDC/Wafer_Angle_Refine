# Derived from NS-DDC/Wafer_Via, upstream commit 5adf92e.
# Source: codex/wafer_notch_angle.py; notch-detection dependency closure only.
# Original detection geometry and defaults are retained; the image-only wrapper
# below preserves the caller's uint8 grayscale, BGR, or BGRA representation.
"""Image-only wafer rotation: ``aligned = angle(src)``.

The wafer notch must be visible near the bottom (90 +/- 45 image-space degrees).
Detection failures raise RuntimeError. Rotation uses the fitted wafer centre,
cubic interpolation, the input canvas size, and zero-filled border pixels.
"""

from __future__ import annotations


import math


from dataclasses import dataclass, field, replace


from pathlib import Path


from typing import Any, Literal, Optional, Tuple, Union


import cv2


import numpy as np


_ImageInput = Union[str, Path, np.ndarray]


_Point = Tuple[float, float]


@dataclass(frozen=True)
class _NotchAngleResult:
    found: bool
    wafer_center_px: _Point
    wafer_radius_px: float
    notch_point_px: _Point
    notch_deepest_point_px: _Point
    notch_angle_deg: float
    reference_angle_deg: float
    correction_angle_deg: float
    notch_depth_px: float
    notch_width_deg: float
    notch_width_px: float
    confidence: float
    radial_noise_px: float
    candidate_arc_px: Tuple[_Point, ...] = field(repr=False)
    wafer_contour_px: np.ndarray
    segmentation_threshold: float
    scale: float
    failure_mode: str
    detection_method: str
    search_center_angle_deg: float
    search_half_width_deg: float
    edge_support: float
    circle_fit_residual_px: float
    roi_center_px: Optional[_Point] = None
    roi_bounds_px: Optional[Tuple[float, float, float, float]] = None
    semicircle_center_px: Optional[_Point] = None
    semicircle_radius_px: Optional[float] = None
    semicircle_radius_x_px: Optional[float] = None
    semicircle_radius_y_px: Optional[float] = None
    semicircle_shape: str = "none"
    semicircle_score: float = 0.0
    semicircle_fit_residual_px: float = 0.0
    background_segmentation_used: bool = False
    background_palette_bgr: Tuple[Tuple[int, int, int], ...] = ()
    background_distance_threshold_lab: float = 0.0
    fallback_attempted: bool = False
    fallback_used: bool = False
    fallback_reason: str = ""
    notch_shoulder_points_px: Optional[Tuple[_Point, _Point]] = None
    fallback_angle_stability_deg: float = 0.0


def _load_bgr(image: _ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError(f"Unsupported image shape: {image.shape}")
    loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if loaded is None:
        raise FileNotFoundError(str(image))
    return loaded


def _normalise_angle(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _lab_edge_strength_from_lab(lab_image: np.ndarray):
    """Return colour-transition strength from a reusable raw LAB image."""

    lab = np.asarray(lab_image)
    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError("lab_image must have shape (height, width, 3).")
    lab = cv2.GaussianBlur(lab, (5, 5), 0).astype(np.float32)
    squared = np.zeros(lab.shape[:2], dtype=np.float32)
    for channel_index in range(3):
        channel = lab[:, :, channel_index]
        gx = cv2.Scharr(channel, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(channel, cv2.CV_32F, 0, 1)
        squared += gx * gx + gy * gy
    edge = np.sqrt(squared).astype(np.float32)
    normaliser = float(np.percentile(edge, 99.5))
    if not np.isfinite(normaliser) or normaliser <= 1e-6:
        raise RuntimeError("Wafer outer edge was not found: image has no usable colour edge.")
    edge = np.clip(edge / normaliser, 0.0, 1.0)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)
    return edge.astype(np.float32), normaliser


def _angle_distance_deg(angles_deg: np.ndarray, centre_deg: float) -> np.ndarray:
    return np.abs((angles_deg - float(centre_deg) + 180.0) % 360.0 - 180.0)


def _polar_sample(
    image: np.ndarray,
    center: _Point,
    radii: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    map_x = (
        float(center[0]) + np.cos(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    map_y = (
        float(center[1]) + np.sin(angles)[:, None] * radii[None, :]
    ).astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _initial_outer_radius(
    edge: np.ndarray,
    center: _Point,
    angles: np.ndarray,
) -> Tuple[float, float]:
    """Find the outer circular edge supported by many unrelated angles."""

    height, width = edge.shape
    max_radius = min(
        float(center[0]),
        float(center[1]),
        float(width - 1) - float(center[0]),
        float(height - 1) - float(center[1]),
    )
    if max_radius < min(height, width) * 0.25:
        raise RuntimeError("Wafer centre hint leaves too little room for an outer circle.")
    radii = np.linspace(max_radius * 0.55, max_radius * 0.995, max(128, int(max_radius * 0.50)))
    polar = _polar_sample(edge, center, radii, angles)
    # A true circumference is present at the same radius over many angles.
    # The 65th percentile rejects isolated die/street edges.
    radial_score = np.percentile(polar, 65.0, axis=0).astype(np.float32)
    radial_score = cv2.GaussianBlur(radial_score.reshape(1, -1), (11, 1), 0).reshape(-1)
    outer_bias = 0.70 + 0.30 * (radii - radii[0]) / max(1e-6, radii[-1] - radii[0])
    scored = radial_score * outer_bias
    index = int(np.argmax(scored))
    return float(radii[index]), float(radial_score[index])


def _track_outer_edge(
    edge: np.ndarray,
    center: _Point,
    radius: float,
    angles: np.ndarray,
    *,
    inward_px: float,
    outward_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track a continuous edge close to the predicted circumference."""

    height, width = edge.shape
    max_corner_radius = math.hypot(width, height)
    radial_count = max(80, int(round(inward_px + outward_px)) * 2)
    radii = np.linspace(
        max(2.0, float(radius) - float(inward_px)),
        min(max_corner_radius, float(radius) + float(outward_px)),
        radial_count,
    )
    polar = _polar_sample(edge, center, radii, angles).astype(np.float32)
    samples_per_degree = len(angles) / 360.0
    angular_kernel = max(3, int(round(0.45 * samples_per_degree)) | 1)
    polar = cv2.GaussianBlur(polar, (3, angular_kernel), 0)

    # Only a weak distance prior is used. At a notch the outer-circle edge is
    # absent, so the actual inner arc must still be allowed to win.
    distance = np.abs(radii - float(radius))
    prior = np.exp(-distance / max(2.0, float(inward_px) * 0.55))
    scored = polar * (0.82 + 0.18 * prior[None, :])
    indices = np.argmax(scored, axis=1)
    boundary = radii[indices].astype(np.float32)
    support = polar[np.arange(len(angles)), indices].astype(np.float32)
    boundary = _circular_median(
        boundary, max(3, int(round(0.25 * samples_per_degree)) | 1)
    )
    return boundary, support


def _fit_circle_from_radial_profile(
    boundary: np.ndarray,
    support: np.ndarray,
    angles: np.ndarray,
    valid: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Fit radius and centre offset from r(theta)=R+dx*cos+dy*sin."""

    design = np.column_stack((np.ones(len(angles)), np.cos(angles), np.sin(angles)))
    base = valid & np.isfinite(boundary) & np.isfinite(support)
    if int(base.sum()) < 60:
        raise RuntimeError("Wafer circle fit has insufficient supported edge angles.")
    support_floor = float(np.percentile(support[base], 35.0))
    base &= support >= support_floor
    keep = base.copy()
    coefficients = np.asarray((float(np.median(boundary[base])), 0.0, 0.0))
    residual_noise = float("inf")
    for _ in range(7):
        weights = np.clip(support[keep], 0.03, 1.0)
        weighted_design = design[keep] * np.sqrt(weights)[:, None]
        weighted_values = boundary[keep] * np.sqrt(weights)
        coefficients = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
        residual = boundary - design @ coefficients
        centre = float(np.median(residual[keep]))
        residual_noise = float(1.4826 * np.median(np.abs(residual[keep] - centre)))
        new_keep = base & (np.abs(residual - centre) <= max(1.25, 3.0 * residual_noise))
        if int(new_keep.sum()) < 60 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
        residual_noise,
    )


def _circular_gaussian(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    half = kernel_size // 2
    extended = np.concatenate((values[-half:], values, values[:half]))
    blurred = cv2.GaussianBlur(
        extended.reshape(1, -1), (kernel_size, 1), 0
    ).reshape(-1)
    return blurred[half:half + len(values)]


def _circular_median(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    half = kernel_size // 2
    extended = np.concatenate((values[-half:], values, values[:half]))
    windows = np.lib.stride_tricks.sliding_window_view(extended, kernel_size)
    return np.median(windows, axis=1).astype(np.float32)


def _circular_candidate_groups(active: np.ndarray):
    if not np.any(active):
        return []
    if np.all(active):
        return [np.arange(len(active), dtype=np.int64)]
    starts = np.flatnonzero(active & ~np.roll(active, 1))
    groups = []
    for start_value in starts:
        start = int(start_value)
        values = [start]
        index = (start + 1) % len(active)
        while active[index] and index != start:
            values.append(index)
            index = (index + 1) % len(active)
        groups.append(np.asarray(values, dtype=np.int64))
    return groups


@dataclass(frozen=True)
class _LocalSemicircleCandidate:
    center: _Point
    radius: float
    score: float
    edge_support: float
    arc_coverage: float
    arc_points: Tuple[_Point, ...]
    roi_bounds: Tuple[int, int, int, int]
    fit_residual: float = 0.0
    radius_x: Optional[float] = None
    radius_y: Optional[float] = None
    shape: str = "semicircle"


@dataclass(frozen=True)
class _RoiBackgroundGeometry:
    palette_lab: np.ndarray = field(repr=False)
    distance_threshold_lab: float
    sample_mask: np.ndarray = field(repr=False)
    background_like_mask: np.ndarray = field(repr=False)
    exterior_background_mask: np.ndarray = field(repr=False)
    wafer_mask: np.ndarray = field(repr=False)
    wafer_contour: np.ndarray = field(repr=False)
    wafer_center: _Point
    wafer_radius: float
    wafer_circle_residual: float
    roi_bounds: Tuple[int, int, int, int]
    outward_unit: _Point


@dataclass(frozen=True)
class _RimIntrusionCandidate:
    angle_deg: float
    depth: float
    width_deg: float
    confidence: float
    arc_points: Tuple[_Point, ...]
    shoulder_points: Tuple[_Point, _Point]
    deepest_point: _Point
    angle_stability_deg: float


def _detect_roi_rim_intrusion(
    geometry: _RoiBackgroundGeometry,
    center: _Point,
    radius: float,
    roi_center: _Point,
    *,
    search_half_width_deg: float,
    radial_inner_ratio: float,
    min_depth: float,
    radius_range: Optional[Tuple[float, float]],
) -> Tuple[Optional[_RimIntrusionCandidate], str]:
    """Shape-free, conservative recovery inside the *same* background ROI.

    All lengths are analysis pixels. Follow each ray from exterior background
    to its first wafer pixel, never through an isolated internal dark feature.
    Use the angular midpoint of two intact mouth shoulders, not the deepest
    pixel. No image reload, new segmentation, circle refit, or exception catch.
    """

    circle_noise = float(geometry.wafer_circle_residual)
    if not math.isfinite(circle_noise) or circle_noise > max(1.5, radius * 0.005):
        return None, "unreliable_wafer_circle"
    mask = geometry.exterior_background_mask
    height, width = mask.shape
    x0, y0, x1, y1 = geometry.roi_bounds
    central_angle = math.atan2(roi_center[1] - center[1], roi_center[0] - center[0])
    half_angle = math.radians(float(search_half_width_deg))
    # A nearly constant arc-pixel spacing avoids resolution-dependent bins.
    count = max(31, int(math.ceil(2.0 * half_angle * radius / 0.6)) + 1)
    theta = np.linspace(central_angle - half_angle, central_angle + half_angle, count)
    arc_step = float((theta[1] - theta[0]) * radius)
    outer_margin = max(5.0, 4.0 * circle_noise, radius * 0.008)
    inner_depth = max(6.0, radius * (1.0 - radial_inner_ratio))
    radial = np.arange(radius + outer_margin, radius - inner_depth, -0.75)
    boundary = np.empty(count, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    # Chunked direct nearest-pixel sampling also works beyond remap's 32767
    # dimension limit, without full angle x depth float64 coordinate arrays.
    for start in range(0, count, 512):
        stop = min(count, start + 512)
        xs = center[0] + np.cos(theta[start:stop])[:, None] * radial[None, :]
        ys = center[1] + np.sin(theta[start:stop])[:, None] * radial[None, :]
        in_bounds = (
            (xs >= max(1, x0 + 1)) & (xs < min(width - 1, x1 - 1))
            & (ys >= max(1, y0 + 1)) & (ys < min(height - 1, y1 - 1))
        )
        xi = np.clip(np.rint(xs).astype(np.int32), 0, width - 1)
        yi = np.clip(np.rint(ys).astype(np.int32), 0, height - 1)
        sampled = (mask[yi, xi] != 0) & in_bounds
        foreground = ~sampled & in_bounds
        first = np.argmax(foreground, axis=1)
        rows = np.arange(stop - start)
        # Every sample up to the boundary must exist, including exterior evidence.
        invalid_prefix = np.cumsum(~in_bounds, axis=1)
        valid[start:stop] = (
            sampled[:, 0] & foreground.any(axis=1) & (first > 0)
            & (invalid_prefix[rows, first] == 0) & (first < len(radial) - 2)
        )
        boundary[start:stop] = radial[first] + 0.375
    if int(valid.sum()) < 24:
        return None, "insufficient_roi_rim_support"
    raw_depth = radius - boundary
    baseline_samples = raw_depth[valid]
    baseline_samples = baseline_samples[
        baseline_samples <= np.percentile(baseline_samples, 45.0)
    ]
    baseline = float(np.median(baseline_samples))
    noise = float(1.4826 * np.median(np.abs(baseline_samples - baseline)))
    if abs(baseline) > max(4.0, circle_noise * 4.0):
        return None, "rim_disagrees_with_wafer_circle"
    profile = np.where(valid, raw_depth - baseline, 0.0).astype(np.float32)
    profile = cv2.medianBlur(profile.reshape(1, -1), 3).ravel().astype(np.float64)
    threshold = max(1.5, noise * 3.0, circle_noise * 2.0)
    # Below the pixel/radial sampling floor, a quantised intact rim can join
    # the mouth during stability trials. Never treat that floor as a cavity.
    mouth_level = max(1.5, threshold * 0.6)
    required_peak = max(float(min_depth), 3.0, threshold * 2.0)
    shoulder_count = max(4, int(math.ceil(4.0 / arc_step)))
    bridge_count = max(1, int(math.floor(2.0 / arc_step)))

    def groups_at(level: float):
        active = valid & (profile > level)
        indices = np.flatnonzero(active)
        for left, right in zip(indices[:-1], indices[1:]):
            if 1 < right - left <= bridge_count + 1 and valid[left:right + 1].all():
                active[left:right + 1] = True
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])
        return list(zip(starts, ends))

    candidates = []
    seen_mouths = set()
    rejected_reason = "no_significant_rim_intrusion"
    for left, right in groups_at(threshold):
        peak_index = int(left + np.argmax(profile[left:right + 1]))
        peak = float(profile[peak_index])
        if peak < required_peak:
            continue
        # Retain shallow mouth flanks as long as they are above rim noise.
        lo, hi = int(left), int(right)
        while lo > 0 and valid[lo - 1] and profile[lo - 1] > mouth_level:
            lo -= 1
        while hi + 1 < count and valid[hi + 1] and profile[hi + 1] > mouth_level:
            hi += 1
        if (lo, hi) in seen_mouths:
            continue
        seen_mouths.add((lo, hi))
        peak_index = int(lo + np.argmax(profile[lo:hi + 1]))
        peak = float(profile[peak_index])
        start, stop = lo - shoulder_count, hi + shoulder_count + 1
        if start < 0 or stop > count or not valid[start:stop].all():
            rejected_reason = "missing_or_clipped_shoulders"
            continue
        flanks = np.r_[profile[start:lo], profile[hi + 1:stop]]
        if np.max(flanks) > threshold or abs(float(np.median(flanks))) > threshold:
            rejected_reason = "shoulders_do_not_return_to_rim"
            continue
        span = float(theta[hi + 1] - theta[lo - 1])
        width_px = float(2.0 * radius * math.sin(span * 0.5))
        if (
            width_px < max(4.0, 4.0 * arc_step)
            or math.degrees(span) > 25.0
            or hi - lo + 1 > int(valid.sum()) * 0.65
        ):
            rejected_reason = "implausible_intrusion_width"
            continue
        if radius_range is not None and not radius_range[0] <= width_px * 0.5 <= radius_range[1]:
            rejected_reason = "intrusion_width_outside_radius_range"
            continue
        mean_depth = float(np.mean(np.maximum(0.0, profile[lo:hi + 1])))
        if mean_depth < max(threshold, peak * 0.12):
            rejected_reason = "insufficient_intrusion_area"
            continue
        angle = float((theta[lo - 1] + theta[hi + 1]) * 0.5)
        # Mouth direction must survive a small change in the noise threshold.
        trial_angles = []
        for multiplier in (0.8, 1.2):
            matching = [
                (a, b) for a, b in groups_at(max(1.5, mouth_level * multiplier))
                if a <= peak_index <= b
            ]
            if len(matching) != 1:
                break
            a, b = matching[0]
            if a < shoulder_count or b + shoulder_count >= count or not valid[a - shoulder_count:b + shoulder_count + 1].all():
                break
            trial_angles.append(float((theta[a - 1] + theta[b + 1]) * 0.5))
        if len(trial_angles) != 2:
            rejected_reason = "unstable_mouth_shoulders"
            continue
        stability = math.degrees(max(abs(value - angle) for value in trial_angles))
        if stability > max(0.25, math.degrees(2.0 / radius)):
            rejected_reason = "unstable_mouth_angle"
            continue
        score = float(np.clip(
            0.5 * min(1.0, peak / (required_peak * 2.0))
            + 0.25 * min(1.0, mean_depth / (threshold * 3.0))
            + 0.25 * max(0.0, 1.0 - stability / max(0.5, math.degrees(4.0 / radius))),
            0.0, 1.0,
        ))
        arc_indices = np.arange(lo - 1, hi + 2)
        arc = tuple(
            (float(center[0] + math.cos(theta[i]) * boundary[i]),
             float(center[1] + math.sin(theta[i]) * boundary[i]))
            for i in arc_indices
        )
        shoulders = tuple(
            (float(center[0] + math.cos(theta[i]) * radius),
             float(center[1] + math.sin(theta[i]) * radius))
            for i in (lo - 1, hi + 1)
        )
        candidates.append(_RimIntrusionCandidate(
            angle_deg=math.degrees(angle) % 360.0,
            depth=max(0.0, float(raw_depth[peak_index])),
            width_deg=math.degrees(span), confidence=score,
            arc_points=arc, shoulder_points=shoulders,
            deepest_point=(float(center[0] + math.cos(theta[peak_index]) * boundary[peak_index]),
                           float(center[1] + math.sin(theta[peak_index]) * boundary[peak_index])),
            angle_stability_deg=stability,
        ))
    if not candidates:
        return None, rejected_reason
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    if len(candidates) > 1 and candidates[1].confidence >= candidates[0].confidence * 0.75:
        return None, "ambiguous_multiple_intrusions"
    return candidates[0], "rim_intrusion_accepted"


def _robust_circle_from_points(
    points: np.ndarray,
    *,
    minimum_points: int = 30,
) -> Tuple[_Point, float, float, np.ndarray]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(values) < int(minimum_points):
        raise RuntimeError("Circle fit has insufficient boundary points.")
    center = np.median(values, axis=0)
    radius = float(np.median(np.linalg.norm(values - center, axis=1)))
    keep = np.ones(len(values), dtype=bool)
    residual = np.zeros(len(values), dtype=np.float64)
    noise = float("inf")
    for _ in range(10):
        selected = values[keep]
        design = np.column_stack((
            2.0 * selected[:, 0],
            2.0 * selected[:, 1],
            np.ones(len(selected)),
        ))
        targets = np.sum(selected * selected, axis=1)
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        center = coefficients[:2]
        radius_squared = float(coefficients[2] + center @ center)
        if radius_squared <= 0.0:
            raise RuntimeError("Circle fit produced a non-positive radius.")
        radius = math.sqrt(radius_squared)
        residual = np.linalg.norm(values - center, axis=1) - radius
        median = float(np.median(residual[keep]))
        noise = float(1.4826 * np.median(np.abs(residual[keep] - median)))
        new_keep = np.abs(residual - median) <= max(1.25, 3.0 * noise)
        if int(new_keep.sum()) < int(minimum_points) or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    fit_residual = float(np.median(np.abs(residual[keep])))
    return (float(center[0]), float(center[1])), float(radius), fit_residual, keep


def _learn_background_from_notch_roi(
    image_bgr: np.ndarray,
    roi_center: _Point,
    roi_half_size: _Point,
    center_hint: _Point,
    *,
    palette_size: int = 3,
    outer_band_fraction: float = 0.28,
    distance_threshold_lab: Optional[float] = None,
    noise_margin_lab: float = 4.0,
    morph_size_px: float = 24.0,
    lab_image: Optional[np.ndarray] = None,
) -> _RoiBackgroundGeometry:
    """Learn exterior colour in the outward ROI band and segment the wafer."""

    height, width = image_bgr.shape[:2]
    x0 = max(0, int(math.floor(roi_center[0] - roi_half_size[0])))
    y0 = max(0, int(math.floor(roi_center[1] - roi_half_size[1])))
    x1 = min(width, int(math.ceil(roi_center[0] + roi_half_size[0])) + 1)
    y1 = min(height, int(math.ceil(roi_center[1] + roi_half_size[1])) + 1)
    if x1 - x0 < 24 or y1 - y0 < 24:
        raise ValueError("notch ROI is too small or lies outside the image.")
    if not 1 <= int(palette_size) <= 8:
        raise ValueError("notch_background_palette_size must be between 1 and 8.")
    if not 0.10 <= float(outer_band_fraction) <= 0.60:
        raise ValueError("notch_background_outer_band_fraction must be in [0.10, 0.60].")

    outward = np.asarray(roi_center, dtype=np.float64) - np.asarray(
        center_hint, dtype=np.float64
    )
    outward_length = float(np.linalg.norm(outward))
    if outward_length <= 1e-6:
        outward = np.asarray((0.0, 1.0), dtype=np.float64)
    else:
        outward /= outward_length

    roi_height, roi_width = y1 - y0, x1 - x0
    local_y, local_x = np.indices((roi_height, roi_width), dtype=np.float32)
    global_x = local_x + float(x0)
    global_y = local_y + float(y0)
    projection = (
        (global_x - float(roi_center[0])) * float(outward[0])
        + (global_y - float(roi_center[1])) * float(outward[1])
    )
    quantile = 1.0 - float(outer_band_fraction)
    projection_threshold = float(np.quantile(projection, quantile))
    sample_local = projection >= projection_threshold
    sample_mask = np.zeros((height, width), dtype=np.uint8)
    sample_mask[y0:y1, x0:x1] = sample_local.astype(np.uint8) * 255

    if lab_image is None:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    else:
        lab = np.asarray(lab_image)
        if lab.shape != image_bgr.shape:
            raise ValueError("lab_image shape must match image_bgr shape.")
        if lab.ndim != 3 or lab.shape[2] != 3:
            raise ValueError("lab_image must have shape (height, width, 3).")
    lab = lab.astype(np.float32, copy=False)
    samples = lab[y0:y1, x0:x1][sample_local].reshape(-1, 3)
    if len(samples) < 64:
        raise RuntimeError("The outward notch ROI band has too few background pixels.")
    stride = max(1, len(samples) // 30000)
    samples_for_fit = samples[::stride].astype(np.float32)
    distinct = np.unique(samples_for_fit.astype(np.uint8), axis=0)
    cluster_count = min(int(palette_size), len(distinct), len(samples_for_fit))
    if cluster_count <= 1:
        palette = np.median(samples_for_fit, axis=0, keepdims=True).astype(np.float32)
        labels = np.zeros((len(samples_for_fit), 1), dtype=np.int32)
    else:
        cv2.setRNGSeed(1907)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            60,
            0.20,
        )
        _, labels, palette = cv2.kmeans(
            samples_for_fit,
            cluster_count,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS,
        )
    assigned = palette[labels.reshape(-1)]
    sample_residual = np.linalg.norm(samples_for_fit - assigned, axis=1)
    automatic_threshold = max(
        8.0,
        float(np.percentile(sample_residual, 98.0)) + float(noise_margin_lab),
    )
    threshold = (
        automatic_threshold
        if distance_threshold_lab is None
        else float(distance_threshold_lab)
    )
    if threshold <= 0.0:
        raise ValueError("notch_background_distance_threshold_lab must be positive.")

    nearest_distance = np.full((height, width), np.inf, dtype=np.float32)
    for colour in palette:
        delta = lab - colour.reshape(1, 1, 3)
        distance = np.sqrt(np.sum(delta * delta, axis=2)).astype(np.float32)
        np.minimum(nearest_distance, distance, out=nearest_distance)
    background_like = (nearest_distance <= threshold).astype(np.uint8) * 255
    morph_size = max(3, int(round(float(morph_size_px))) | 1)
    morph_size = min(morph_size, max(3, (min(height, width) // 12) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    background_like = cv2.morphologyEx(
        background_like, cv2.MORPH_CLOSE, kernel
    )

    component_count, components, stats, _ = cv2.connectedComponentsWithStats(
        (background_like > 0).astype(np.uint8), 8
    )
    border_labels = np.unique(np.concatenate((
        components[0, :],
        components[-1, :],
        components[:, 0],
        components[:, -1],
    )))
    border_labels = border_labels[border_labels > 0]
    if not len(border_labels):
        raise RuntimeError("ROI background colour did not connect to the image border.")
    exterior_label = int(max(
        border_labels, key=lambda label: int(stats[int(label), cv2.CC_STAT_AREA])
    ))
    exterior_background = (components == exterior_label).astype(np.uint8) * 255

    foreground = (exterior_background == 0).astype(np.uint8)
    open_size = max(3, int(round(float(morph_size_px) * 0.45)) | 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, open_kernel)
    fg_count, fg_components, fg_stats, fg_centroids = cv2.connectedComponentsWithStats(
        foreground, 8
    )
    center_x = int(np.clip(round(center_hint[0]), 0, width - 1))
    center_y = int(np.clip(round(center_hint[1]), 0, height - 1))
    wafer_label = int(fg_components[center_y, center_x])
    if wafer_label <= 0:
        candidates = []
        for label in range(1, fg_count):
            area = int(fg_stats[label, cv2.CC_STAT_AREA])
            centroid = fg_centroids[label]
            distance_to_hint = float(np.linalg.norm(centroid - np.asarray(center_hint)))
            candidates.append((area / max(1.0, 1.0 + distance_to_hint), label))
        if not candidates:
            raise RuntimeError("Wafer component was not found after ROI background segmentation.")
        wafer_label = int(max(candidates)[1])
    wafer_mask = (fg_components == wafer_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        wafer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("Wafer contour was not found after ROI background segmentation.")
    wafer_contour = max(contours, key=cv2.contourArea)
    contour_values = wafer_contour.reshape(-1, 2)
    border_clear = (
        (contour_values[:, 0] > 1)
        & (contour_values[:, 0] < width - 2)
        & (contour_values[:, 1] > 1)
        & (contour_values[:, 1] < height - 2)
    )
    fit_points = contour_values[border_clear]
    if len(fit_points) > 6000:
        fit_points = fit_points[::max(1, len(fit_points) // 6000)]
    wafer_center, wafer_radius, circle_residual, _ = _robust_circle_from_points(
        fit_points, minimum_points=100
    )
    if wafer_radius <= min(height, width) * 0.20:
        raise RuntimeError("Background-segmented wafer radius is implausibly small.")

    return _RoiBackgroundGeometry(
        palette_lab=palette.astype(np.float32),
        distance_threshold_lab=float(threshold),
        sample_mask=sample_mask,
        background_like_mask=background_like,
        exterior_background_mask=exterior_background,
        wafer_mask=wafer_mask,
        wafer_contour=wafer_contour,
        wafer_center=wafer_center,
        wafer_radius=float(wafer_radius),
        wafer_circle_residual=float(circle_residual),
        roi_bounds=(x0, y0, x1, y1),
        outward_unit=(float(outward[0]), float(outward[1])),
    )


def _sample_fitted_arc_support(
    edge: np.ndarray,
    arc_points: np.ndarray,
) -> Tuple[float, float, float]:
    """Measure local edge support and left/right balance along a fitted arc."""

    height, width = edge.shape
    values = []
    for point in np.asarray(arc_points, dtype=np.float64).reshape(-1, 2):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x0, x1 = max(0, x - 2), min(width, x + 3)
        y0, y1 = max(0, y - 2), min(height, y + 3)
        values.append(
            0.0 if x0 >= x1 or y0 >= y1 else float(np.max(edge[y0:y1, x0:x1]))
        )
    support = np.asarray(values, dtype=np.float64)
    if not len(support):
        return 0.0, 0.0, 0.0
    edge_support = float(np.mean(support))
    support_floor = max(0.08, float(np.percentile(support, 35.0)) * 0.70)
    coverage = float(np.mean(support >= support_floor))
    midpoint = len(support) // 2
    left = float(np.mean(support[:midpoint])) if midpoint else edge_support
    right = (
        float(np.mean(support[midpoint + 1 :]))
        if midpoint + 1 < len(support) else edge_support
    )
    symmetry = min(left, right) / max(1e-6, max(left, right))
    return edge_support, coverage, float(symmetry)


def _fit_semiellipse_from_background_boundary(
    geometry: _RoiBackgroundGeometry,
    roi_center: _Point,
    roi_half_size: _Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Fit a shallow/wide semi-ellipse to the exterior intrusion boundary."""

    contours, _ = cv2.findContours(
        geometry.exterior_background_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours], axis=0)
    x0, y0, x1, y1 = geometry.roi_bounds
    in_roi = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    points = points[in_roi].astype(np.float64)
    if len(points) < 24:
        return None

    wafer_center = np.asarray(geometry.wafer_center, dtype=np.float64)
    outward = np.asarray(geometry.outward_unit, dtype=np.float64)
    outward /= max(1e-9, float(np.linalg.norm(outward)))
    tangent = np.asarray((-outward[1], outward[0]), dtype=np.float64)
    vectors = points - wafer_center
    radial_distance = np.linalg.norm(vectors, axis=1)
    depth = float(geometry.wafer_radius) - radial_distance
    tangential = vectors @ tangent
    outward_projection = vectors @ outward
    minimum_half_size = min(float(roi_half_size[0]), float(roi_half_size[1]))
    if radius_range is None:
        min_half_width = max(3.0, minimum_half_size * 0.035)
        max_half_width = max(min_half_width + 2.0, minimum_half_size * 0.55)
    else:
        min_half_width, max_half_width = (
            float(radius_range[0]), float(radius_range[1])
        )
        if min_half_width <= 0.0 or max_half_width <= min_half_width:
            raise ValueError(
                "notch_semicircle_radius_range_px must be (positive_min, larger_max)."
            )

    noise_floor = max(0.75, float(geometry.wafer_circle_residual) * 1.35)
    usable = (
        (depth >= noise_floor)
        & (depth <= max(4.0, minimum_half_size * 0.75))
        & (outward_projection >= float(geometry.wafer_radius) - 2.2 * minimum_half_size)
        & (np.abs(tangential - float(np.dot(np.asarray(roi_center) - wafer_center, tangent)))
           <= float(roi_half_size[0]) * 1.10)
    )
    tangential = tangential[usable]
    depth = depth[usable]
    if len(depth) < 24:
        return None

    # Keep the deepest exterior-boundary sample in each tangential pixel bin.
    bins = np.rint(tangential).astype(np.int32)
    unique_bins = np.unique(bins)
    fitted_t = []
    fitted_d = []
    for value in unique_bins:
        selected_depth = depth[bins == value]
        fitted_t.append(float(value))
        fitted_d.append(float(np.max(selected_depth)))
    fitted_t = np.asarray(fitted_t, dtype=np.float64)
    fitted_d = np.asarray(fitted_d, dtype=np.float64)
    if len(fitted_t) < 18:
        return None

    order = np.argsort(fitted_t)
    fitted_t, fitted_d = fitted_t[order], fitted_d[order]
    expected_t = float(np.dot(np.asarray(roi_center) - wafer_center, tangent))
    central = np.abs(fitted_t - expected_t) <= float(roi_half_size[0]) * 0.75
    if not np.any(central):
        return None
    local_peak = float(np.max(fitted_d[central]))
    strong_threshold = max(noise_floor * 1.55, local_peak * 0.12)
    strong_indices = np.flatnonzero(central & (fitted_d >= strong_threshold))
    if len(strong_indices) < 8:
        return None
    split_at = np.flatnonzero(np.diff(fitted_t[strong_indices]) > 3.5) + 1
    groups = np.split(strong_indices, split_at)
    groups = [group for group in groups if len(group) >= 8]
    if not groups:
        return None
    group = max(
        groups,
        key=lambda values: float(np.sum(fitted_d[values]))
        * math.exp(
            -0.5
            * (
                (float(np.mean(fitted_t[values])) - expected_t)
                / max(3.0, float(roi_half_size[0]) * 0.30)
            )
            ** 2
        ),
    )
    lower = float(fitted_t[group[0]] - 4.0)
    upper = float(fitted_t[group[-1]] + 4.0)
    selected_group = (fitted_t >= lower) & (fitted_t <= upper)
    fitted_t, fitted_d = fitted_t[selected_group], fitted_d[selected_group]
    if len(fitted_t) < 18:
        return None

    # Semi-ellipse linearisation: d^2 = A*t^2 + B*t + C.  Robust iterations
    # reject texture/noise points while retaining the broad, shallow arc.
    origin_t = float(np.median(fitted_t))
    x = fitted_t - origin_t
    keep = np.ones(len(x), dtype=bool)
    half_width = depth_axis = center_offset = fit_residual = 0.0
    for _ in range(10):
        if int(keep.sum()) < 18:
            return None
        design = np.column_stack((x[keep] * x[keep], x[keep], np.ones(int(keep.sum()))))
        coefficients = np.linalg.lstsq(
            design, fitted_d[keep] * fitted_d[keep], rcond=None
        )[0]
        quadratic, linear, constant = (float(value) for value in coefficients)
        if quadratic >= -1e-8:
            return None
        center_local = -linear / (2.0 * quadratic)
        depth_squared = constant - quadratic * center_local * center_local
        if depth_squared <= 0.0:
            return None
        depth_axis = math.sqrt(depth_squared)
        half_width_squared = -depth_squared / quadratic
        if half_width_squared <= 0.0:
            return None
        half_width = math.sqrt(half_width_squared)
        center_offset = origin_t + center_local
        normalized = (fitted_t - center_offset) / max(half_width, 1e-6)
        predicted = depth_axis * np.sqrt(
            np.maximum(0.0, 1.0 - normalized * normalized)
        )
        residual = fitted_d - predicted
        valid_span = np.abs(normalized) <= 1.08
        current = keep & valid_span
        if int(current.sum()) < 18:
            return None
        median = float(np.median(residual[current]))
        noise = float(
            1.4826 * np.median(np.abs(residual[current] - median))
        )
        new_keep = valid_span & (
            np.abs(residual - median) <= max(1.15, 2.8 * noise)
        )
        if int(new_keep.sum()) < 18:
            return None
        fit_residual = float(np.median(np.abs(residual[new_keep])))
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    if not (min_half_width * 0.70 <= half_width <= max_half_width * 1.25):
        return None
    aspect = depth_axis / max(half_width, 1e-6)
    if not 0.12 <= aspect <= 1.60:
        return None

    direction_vector = outward * float(geometry.wafer_radius) + tangent * center_offset
    direction_length = float(np.linalg.norm(direction_vector))
    if direction_length <= 1e-6:
        return None
    direction = direction_vector / direction_length
    arc_tangent = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    inward = -direction
    baseline_center = wafer_center + direction * float(geometry.wafer_radius)
    unit = np.linspace(-1.0, 1.0, 97, dtype=np.float64)
    arc_values = (
        baseline_center[None, :]
        + arc_tangent[None, :] * (unit * half_width)[:, None]
        + inward[None, :] * (
            depth_axis * np.sqrt(np.maximum(0.0, 1.0 - unit * unit))
        )[:, None]
    )
    boundary_edge = cv2.morphologyEx(
        geometry.exterior_background_mask,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(np.float32) / 255.0
    edge_support, arc_coverage, symmetry = _sample_fitted_arc_support(
        boundary_edge, arc_values
    )
    apex = baseline_center + inward * depth_axis
    center_distance = float(np.linalg.norm(apex - np.asarray(roi_center)))
    center_prior = math.exp(
        -0.5 * (center_distance / max(4.0, minimum_half_size * 0.28)) ** 2
    )
    fit_quality = math.exp(
        -fit_residual / max(1.0, depth_axis * 0.10)
    )
    score = float(np.clip(
        0.26 * edge_support
        + 0.18 * arc_coverage
        + 0.10 * symmetry
        + 0.18 * center_prior
        + 0.28 * fit_quality,
        0.0,
        1.0,
    ))
    stride = max(1, len(arc_values) // 48)
    return _LocalSemicircleCandidate(
        center=(float(baseline_center[0]), float(baseline_center[1])),
        radius=float(half_width),
        score=score,
        edge_support=float(edge_support),
        arc_coverage=float(arc_coverage),
        arc_points=tuple(
            (float(point[0]), float(point[1])) for point in arc_values[::stride]
        ),
        roi_bounds=geometry.roi_bounds,
        fit_residual=float(fit_residual),
        radius_x=float(half_width),
        radius_y=float(depth_axis),
        shape="semiellipse",
    )


def _fit_circle_from_background_boundary(
    geometry: _RoiBackgroundGeometry,
    roi_center: _Point,
    roi_half_size: _Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Fit the exterior-background intrusion contour around the expected notch."""

    contours, _ = cv2.findContours(
        geometry.exterior_background_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours], axis=0)
    x0, y0, x1, y1 = geometry.roi_bounds
    in_roi = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    points = points[in_roi]
    if len(points) < 20:
        return None

    minimum_half_size = min(float(roi_half_size[0]), float(roi_half_size[1]))
    if radius_range is None:
        min_radius = max(3.0, minimum_half_size * 0.035)
        max_radius = max(min_radius + 2.0, minimum_half_size * 0.55)
    else:
        min_radius, max_radius = float(radius_range[0]), float(radius_range[1])
    relative = points.astype(np.float64) - np.asarray(roi_center, dtype=np.float64)
    distances = np.linalg.norm(relative, axis=1)
    inward = -np.asarray(geometry.outward_unit, dtype=np.float64)
    inward_projection = relative @ inward
    usable = (
        (distances >= min_radius)
        & (distances <= max_radius)
        & (inward_projection >= -max(2.0, min_radius * 0.25))
    )
    points = points[usable]
    distances = distances[usable]
    if len(points) < 20:
        return None

    bin_width = max(1.0, minimum_half_size / 240.0)
    bins = np.arange(min_radius, max_radius + 2.0 * bin_width, bin_width)
    histogram, edges = np.histogram(distances, bins=bins)
    if not np.any(histogram):
        return None
    smooth_histogram = cv2.GaussianBlur(
        histogram.astype(np.float32).reshape(1, -1), (5, 1), 0
    ).reshape(-1)
    peak_indices = np.argsort(smooth_histogram)[::-1][:12]
    boundary_edge = cv2.morphologyEx(
        geometry.exterior_background_mask,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(np.float32) / 255.0
    best: Optional[_LocalSemicircleCandidate] = None
    for peak_index in peak_indices:
        peak_radius = float((edges[peak_index] + edges[peak_index + 1]) * 0.5)
        band = max(2.0, peak_radius * 0.08)
        selected = points[np.abs(distances - peak_radius) <= band]
        if len(selected) < 20:
            continue
        try:
            center, radius, fit_residual, _ = _robust_circle_from_points(
                selected, minimum_points=18
            )
        except RuntimeError:
            continue
        center_distance = float(np.linalg.norm(
            np.asarray(center) - np.asarray(roi_center)
        ))
        if center_distance > minimum_half_size * 0.45:
            continue
        if not min_radius * 0.70 <= radius <= max_radius * 1.20:
            continue
        inward_angle = math.atan2(
            geometry.wafer_center[1] - center[1],
            geometry.wafer_center[0] - center[0],
        )
        edge_support, arc_coverage, symmetry, arc_points = _sample_semicircle_support(
            boundary_edge, center, radius, inward_angle
        )
        center_prior = math.exp(
            -0.5 * (center_distance / max(4.0, minimum_half_size * 0.20)) ** 2
        )
        fit_quality = math.exp(
            -fit_residual / max(1.25, radius * 0.055)
        )
        score = float(np.clip(
            0.24 * edge_support
            + 0.16 * arc_coverage
            + 0.10 * symmetry
            + 0.25 * center_prior
            + 0.25 * fit_quality,
            0.0,
            1.0,
        ))
        candidate = _LocalSemicircleCandidate(
            center=center,
            radius=radius,
            score=score,
            edge_support=edge_support,
            arc_coverage=arc_coverage,
            arc_points=arc_points,
            roi_bounds=geometry.roi_bounds,
            fit_residual=fit_residual,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def _fit_semicircle_from_background_boundary(
    geometry: _RoiBackgroundGeometry,
    roi_center: _Point,
    roi_half_size: _Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Fit a semi-ellipse first, then retain the historical circle fallback."""

    candidate = _fit_semiellipse_from_background_boundary(
        geometry, roi_center, roi_half_size, radius_range
    )
    if candidate is not None:
        return candidate
    return _fit_circle_from_background_boundary(
        geometry, roi_center, roi_half_size, radius_range
    )


def _normalise_roi_half_size(
    value: Union[float, Tuple[float, float]],
    *,
    scale: float,
) -> _Point:
    if isinstance(value, (int, float, np.integer, np.floating)):
        half_width = half_height = float(value)
    else:
        if len(value) != 2:
            raise ValueError("notch_roi_half_size_px must be a number or (half_width, half_height).")
        half_width, half_height = float(value[0]), float(value[1])
    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("notch_roi_half_size_px values must be positive.")
    return half_width * float(scale), half_height * float(scale)


def _sample_semicircle_support(
    edge: np.ndarray,
    center: _Point,
    radius: float,
    inward_angle_rad: float,
) -> Tuple[float, float, float, Tuple[_Point, ...]]:
    """Measure the inward-facing half of a local U-shaped notch circle."""

    arc_angles = np.linspace(
        float(inward_angle_rad) - math.radians(100.0),
        float(inward_angle_rad) + math.radians(100.0),
        161,
        dtype=np.float64,
    )
    band = max(1.0, min(5.0, float(radius) * 0.10))
    radial_offsets = np.linspace(-band, band, 7, dtype=np.float64)
    radii = np.maximum(1.0, float(radius) + radial_offsets)
    sampled = _polar_sample(edge, center, radii, arc_angles)
    supported = sampled.max(axis=1)
    edge_support = float(np.mean(supported))
    support_floor = max(0.08, float(np.percentile(supported, 35.0)) * 0.70)
    arc_coverage = float(np.mean(supported >= support_floor))
    midpoint = len(supported) // 2
    left_support = float(np.mean(supported[:midpoint]))
    right_support = float(np.mean(supported[midpoint + 1:]))
    symmetry = min(left_support, right_support) / max(
        1e-6, max(left_support, right_support)
    )
    stride = max(1, len(arc_angles) // 48)
    arc_points = tuple(
        (
            float(center[0] + math.cos(float(angle)) * float(radius)),
            float(center[1] + math.sin(float(angle)) * float(radius)),
        )
        for angle in arc_angles[::stride]
    )
    return edge_support, arc_coverage, float(symmetry), arc_points


def _refine_semicircle_candidate(
    edge: np.ndarray,
    wafer_center: _Point,
    candidate: _LocalSemicircleCandidate,
) -> _LocalSemicircleCandidate:
    """Robustly re-fit the actual inward arc after coarse Hough detection."""

    initial_center = np.asarray(candidate.center, dtype=np.float64)
    initial_radius = float(candidate.radius)
    inward_angle = math.atan2(
        float(wafer_center[1]) - float(initial_center[1]),
        float(wafer_center[0]) - float(initial_center[0]),
    )
    arc_angles = np.linspace(
        inward_angle - math.radians(100.0),
        inward_angle + math.radians(100.0),
        241,
        dtype=np.float64,
    )
    radii = np.linspace(
        max(2.0, initial_radius * 0.65), initial_radius * 1.35, 101
    )
    polar = _polar_sample(edge, candidate.center, radii, arc_angles)
    indices = np.argmax(polar, axis=1)
    support = polar[np.arange(len(arc_angles)), indices]
    selected_radii = radii[indices]
    support_floor = max(0.08, float(np.percentile(support, 35.0)))
    keep = (
        (support >= support_floor)
        & (np.abs(selected_radii - initial_radius) <= initial_radius * 0.30)
    )
    points = np.column_stack((
        initial_center[0] + np.cos(arc_angles) * selected_radii,
        initial_center[1] + np.sin(arc_angles) * selected_radii,
    ))
    if int(keep.sum()) < 30:
        return candidate

    fitted_center = initial_center.copy()
    fitted_radius = initial_radius
    residual = np.zeros(len(points), dtype=np.float64)
    for _ in range(8):
        selected = points[keep]
        weights = np.clip(support[keep], 0.03, 1.0)
        design = np.column_stack((
            2.0 * selected[:, 0],
            2.0 * selected[:, 1],
            np.ones(len(selected)),
        ))
        values = np.sum(selected * selected, axis=1)
        root_weights = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            design * root_weights[:, None], values * root_weights, rcond=None
        )[0]
        fitted_center = coefficients[:2]
        radius_squared = float(coefficients[2] + fitted_center @ fitted_center)
        if radius_squared <= 0.0:
            return candidate
        fitted_radius = math.sqrt(radius_squared)
        residual = np.linalg.norm(points - fitted_center, axis=1) - fitted_radius
        median = float(np.median(residual[keep]))
        noise = float(1.4826 * np.median(np.abs(residual[keep] - median)))
        refined_keep = keep & (np.abs(residual - median) <= max(1.25, 2.5 * noise))
        if int(refined_keep.sum()) < 30 or np.array_equal(refined_keep, keep):
            break
        keep = refined_keep

    center_shift = float(np.linalg.norm(fitted_center - initial_center))
    fit_residual = float(np.median(np.abs(residual[keep])))
    reliable = bool(
        center_shift <= initial_radius * 0.45
        and initial_radius * 0.55 <= fitted_radius <= initial_radius * 1.55
        and fit_residual <= max(1.5, fitted_radius * 0.06)
    )
    if not reliable:
        return replace(candidate, fit_residual=fit_residual)

    refined_inward_angle = math.atan2(
        float(wafer_center[1]) - float(fitted_center[1]),
        float(wafer_center[0]) - float(fitted_center[0]),
    )
    edge_support, arc_coverage, _, arc_points = _sample_semicircle_support(
        edge,
        (float(fitted_center[0]), float(fitted_center[1])),
        fitted_radius,
        refined_inward_angle,
    )
    return replace(
        candidate,
        center=(float(fitted_center[0]), float(fitted_center[1])),
        radius=float(fitted_radius),
        edge_support=float(edge_support),
        arc_coverage=float(arc_coverage),
        arc_points=arc_points,
        fit_residual=fit_residual,
    )


def _detect_semicircle_in_roi(
    edge: np.ndarray,
    wafer_center: _Point,
    wafer_radius: float,
    roi_center: _Point,
    roi_half_size: _Point,
    radius_range: Optional[Tuple[float, float]],
) -> Optional[_LocalSemicircleCandidate]:
    """Find a small inward-facing semicircle only inside a user ROI."""

    height, width = edge.shape
    x0 = max(0, int(math.floor(float(roi_center[0]) - float(roi_half_size[0]))))
    y0 = max(0, int(math.floor(float(roi_center[1]) - float(roi_half_size[1]))))
    x1 = min(width, int(math.ceil(float(roi_center[0]) + float(roi_half_size[0]))) + 1)
    y1 = min(height, int(math.ceil(float(roi_center[1]) + float(roi_half_size[1]))) + 1)
    if x1 - x0 < 24 or y1 - y0 < 24:
        raise ValueError("notch ROI is too small or lies outside the image.")

    minimum_half_size = min(float(roi_half_size[0]), float(roi_half_size[1]))
    if radius_range is None:
        min_radius = max(3.0, minimum_half_size * 0.035)
        max_radius = max(min_radius + 2.0, minimum_half_size * 0.55)
    else:
        min_radius, max_radius = float(radius_range[0]), float(radius_range[1])
        if min_radius <= 0.0 or max_radius <= min_radius:
            raise ValueError(
                "notch_semicircle_radius_range_px must be (positive_min, larger_max)."
            )

    crop = np.clip(edge[y0:y1, x0:x1] * 255.0, 0.0, 255.0).astype(np.uint8)
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    circles = cv2.HoughCircles(
        crop,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(4.0, min_radius * 0.70),
        param1=40.0,
        param2=max(6.0, min(crop.shape) * 0.018),
        minRadius=max(2, int(math.floor(min_radius))),
        maxRadius=max(3, int(math.ceil(max_radius))),
    )
    if circles is None:
        return None

    wafer_center_array = np.asarray(wafer_center, dtype=np.float64)
    roi_center_array = np.asarray(roi_center, dtype=np.float64)
    # The caller supplies this coordinate precisely because the hardware keeps
    # the notch in a stable area. Give proximity real weight after the arc has
    # passed the geometric filters; otherwise a stronger decorative/internal
    # circle elsewhere in a large ROI can still win.
    roi_scale = max(4.0, 0.20 * minimum_half_size)
    candidate_pool: List[_LocalSemicircleCandidate] = []
    for local_x, local_y, candidate_radius in circles[0]:
        center = np.asarray(
            (float(local_x) + float(x0), float(local_y) + float(y0)),
            dtype=np.float64,
        )
        radius = float(candidate_radius)
        center_radius = float(np.linalg.norm(center - wafer_center_array))
        ring_error = abs(center_radius - float(wafer_radius))
        if ring_error > max(radius * 2.0, minimum_half_size * 0.20):
            continue

        inward_angle = math.atan2(
            float(wafer_center[1]) - float(center[1]),
            float(wafer_center[0]) - float(center[0]),
        )
        edge_support, arc_coverage, symmetry, arc_points = _sample_semicircle_support(
            edge, (float(center[0]), float(center[1])), radius, inward_angle
        )
        hint_distance = float(np.linalg.norm(center - roi_center_array))
        center_prior = math.exp(-0.5 * (hint_distance / roi_scale) ** 2)
        ring_prior = math.exp(
            -0.5 * (ring_error / max(2.0, radius * 0.90)) ** 2
        )
        score = float(np.clip(
            0.32 * edge_support
            + 0.16 * arc_coverage
            + 0.12 * symmetry
            + 0.32 * center_prior
            + 0.08 * ring_prior,
            0.0,
            1.0,
        ))
        candidate = _LocalSemicircleCandidate(
            center=(float(center[0]), float(center[1])),
            radius=radius,
            score=score,
            edge_support=edge_support,
            arc_coverage=arc_coverage,
            arc_points=arc_points,
            roi_bounds=(x0, y0, x1, y1),
        )
        candidate_pool.append(candidate)
    if not candidate_pool:
        return None

    # Refine several strong coarse candidates. Decorative circles can have a
    # slightly higher Hough score, while the true notch wins decisively after
    # its inward arc is fitted with a low radial residual.
    best: Optional[_LocalSemicircleCandidate] = None
    for coarse in sorted(
        candidate_pool, key=lambda item: item.score, reverse=True
    )[:20]:
        refined = _refine_semicircle_candidate(edge, wafer_center, coarse)
        fit_quality = (
            math.exp(
                -float(refined.fit_residual)
                / max(1.5, float(refined.radius) * 0.06)
            )
            if refined.fit_residual > 0.0
            else 0.0
        )
        refined = replace(
            refined,
            score=float(np.clip(0.80 * coarse.score + 0.20 * fit_quality, 0.0, 1.0)),
        )
        if best is None or refined.score > best.score:
            best = refined
    return best


def _detect_wafer_notch(
    image: _ImageInput,
    *,
    reference_angle_deg: float = 90.0,
    max_dimension: int = 3072,
    angle_samples: int = 3600,
    baseline_window_deg: float = 10.0,
    radial_inner_ratio: float = 0.85,
    min_notch_depth_px: Optional[float] = None,
    min_notch_depth_ratio: float = 0.001,
    min_wide_notch_deg: float = 2.0,
    search_center_angle_deg: float = 90.0,
    search_half_width_deg: float = 45.0,
    wafer_center_hint_px: Optional[_Point] = None,
    wafer_radius_hint_px: Optional[float] = None,
    notch_roi_center_px: Optional[_Point] = None,
    notch_roi_half_size_px: Union[float, Tuple[float, float]] = 600.0,
    notch_semicircle_radius_range_px: Optional[Tuple[float, float]] = None,
    notch_semicircle_min_score: float = 0.55,
    notch_use_roi_background: bool = True,
    notch_background_palette_size: int = 3,
    notch_background_outer_band_fraction: float = 0.28,
    notch_background_distance_threshold_lab: Optional[float] = None,
    notch_background_noise_margin_lab: float = 4.0,
    notch_background_morph_px: float = 24.0,
    notch_fallback_mode: Literal["rim_intrusion", "none"] = "rim_intrusion",
    failure_mode: Literal["error", "zero"] = "error",
    require_notch: Optional[bool] = None,
) -> _NotchAngleResult:
    """Find a local inward deviation of the wafer's geometric outer edge.

    ``notch_angle_deg`` uses image coordinates (right=0, down=90). The returned
    ``correction_angle_deg`` is suitable for ``cv2.getRotationMatrix2D`` and
    moves the detected notch to ``reference_angle_deg``.

    Without an ROI, LAB colour-gradient magnitude supplies edge evidence and
    no foreground/background colour is assumed. With
    ``notch_roi_center_px=(x, y)`` the default ROI mode learns the wafer-exterior
    background palette from the outward part of that crop. Only the
    border-connected background is retained; its boundary supplies both a
    noise-resistant wafer silhouette and the local inward semicircle or
    wide/shallow semi-ellipse.
    ``notch_roi_half_size_px`` is a scalar or ``(half_width, half_height)`` in
    full-resolution pixels. ``notch_semicircle_radius_range_px`` and
    ``notch_background_morph_px`` are also full-resolution pixel values.

    ``failure_mode="error"`` raises when no notch is reliable.
    ``failure_mode="zero"`` returns ``found=False`` and a zero correction.
    ``require_notch`` remains as a backwards-compatible alias.
    ``notch_fallback_mode="rim_intrusion"`` retries rejected ROI shape fits
    using the same border-connected background and circle. It never changes a
    successful primary result, and never catches unrelated exceptions. Use
    ``"none"`` to disable it. Without a background ROI this retry is inapplicable.
    """

    mode = str(failure_mode).strip().lower()
    if require_notch is not None:
        mode = "error" if bool(require_notch) else "zero"
    if mode not in ("error", "zero"):
        raise ValueError("failure_mode must be 'error' or 'zero'.")
    fallback_mode = str(notch_fallback_mode).strip().lower()
    if fallback_mode not in ("rim_intrusion", "none"):
        raise ValueError("notch_fallback_mode must be 'rim_intrusion' or 'none'.")
    if not 2.0 <= float(search_half_width_deg) <= 120.0:
        raise ValueError("search_half_width_deg must be between 2 and 120 degrees.")
    if not 0.50 <= float(radial_inner_ratio) < 1.0:
        raise ValueError("radial_inner_ratio must be in [0.50, 1.0).")
    if not 0.0 <= float(notch_semicircle_min_score) <= 1.0:
        raise ValueError("notch_semicircle_min_score must be between 0 and 1.")
    if float(notch_background_noise_margin_lab) < 0.0:
        raise ValueError("notch_background_noise_margin_lab must be non-negative.")
    if float(notch_background_morph_px) <= 0.0:
        raise ValueError("notch_background_morph_px must be positive.")

    source = _load_bgr(image)
    full_height, full_width = source.shape[:2]
    scale = min(1.0, float(max_dimension) / max(full_height, full_width))
    if scale < 1.0:
        work = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        work = source

    # Convert once. Edge extraction uses a blurred copy while ROI background
    # learning uses the unblurred LAB values for colour-distance segmentation.
    work_lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    edge, edge_normaliser = _lab_edge_strength_from_lab(work_lab)
    work_height, work_width = work.shape[:2]

    angle_samples = max(720, int(angle_samples))
    angles = np.arange(angle_samples, dtype=np.float64) * (
        2.0 * math.pi / angle_samples
    )
    angles_deg = np.degrees(angles)
    if wafer_center_hint_px is None:
        center = (work_width / 2.0, work_height / 2.0)
    else:
        center = (
            float(wafer_center_hint_px[0]) * scale,
            float(wafer_center_hint_px[1]) * scale,
        )
    roi_center: Optional[_Point] = None
    roi_half_size: Optional[_Point] = None
    semicircle_radius_range: Optional[Tuple[float, float]] = None
    background_geometry: Optional[_RoiBackgroundGeometry] = None
    effective_search_center_angle_deg = float(search_center_angle_deg) % 360.0
    if notch_roi_center_px is not None:
        roi_center = (
            float(notch_roi_center_px[0]) * scale,
            float(notch_roi_center_px[1]) * scale,
        )
        if not (
            -0.05 * work_width <= roi_center[0] <= 1.05 * work_width
            and -0.05 * work_height <= roi_center[1] <= 1.05 * work_height
        ):
            raise ValueError("notch_roi_center_px lies outside the input image.")
        roi_half_size = _normalise_roi_half_size(
            notch_roi_half_size_px, scale=scale
        )
        effective_search_center_angle_deg = float(
            math.degrees(
                math.atan2(roi_center[1] - center[1], roi_center[0] - center[0])
            )
            % 360.0
        )
        if notch_semicircle_radius_range_px is not None:
            semicircle_radius_range = (
                float(notch_semicircle_radius_range_px[0]) * scale,
                float(notch_semicircle_radius_range_px[1]) * scale,
            )
        if bool(notch_use_roi_background):
            background_geometry = _learn_background_from_notch_roi(
                work,
                roi_center,
                roi_half_size,
                center,
                palette_size=notch_background_palette_size,
                outer_band_fraction=notch_background_outer_band_fraction,
                distance_threshold_lab=notch_background_distance_threshold_lab,
                noise_margin_lab=notch_background_noise_margin_lab,
                morph_size_px=float(notch_background_morph_px) * scale,
                lab_image=work_lab,
            )
            if wafer_center_hint_px is None:
                center = background_geometry.wafer_center
    del work_lab
    search_distance = _angle_distance_deg(
        angles_deg, effective_search_center_angle_deg
    )
    search_mask = search_distance <= float(search_half_width_deg)
    fit_mask = search_distance >= float(search_half_width_deg) + 5.0

    if wafer_radius_hint_px is None:
        if background_geometry is not None:
            radius = float(background_geometry.wafer_radius)
        else:
            radius, _ = _initial_outer_radius(edge, center, angles)
    else:
        radius = float(wafer_radius_hint_px) * scale
    if radius <= min(work_height, work_width) * 0.20:
        raise RuntimeError("Estimated wafer radius is implausibly small.")

    circle_fit_noise = (
        float(background_geometry.wafer_circle_residual)
        if background_geometry is not None else float("inf")
    )
    # Re-centre using the first harmonic of the tracked radius. The notch
    # sector is excluded, so a wide or deep notch cannot pull the fitted circle.
    for _ in range(0 if background_geometry is not None else 4):
        fit_window = max(12.0, radius * 0.08)
        boundary, support = _track_outer_edge(
            edge,
            center,
            radius,
            angles,
            inward_px=fit_window,
            outward_px=fit_window,
        )
        fitted_radius, offset_x, offset_y, circle_fit_noise = _fit_circle_from_radial_profile(
            boundary, support, angles, fit_mask
        )
        max_step = max(3.0, radius * 0.06)
        offset_length = math.hypot(offset_x, offset_y)
        if offset_length > max_step:
            factor = max_step / offset_length
            offset_x *= factor
            offset_y *= factor
        center = (center[0] + offset_x, center[1] + offset_y)
        radius = float(fitted_radius)
        if not np.isfinite(radius) or radius <= min(work_height, work_width) * 0.20:
            raise RuntimeError("Wafer circle fit produced an invalid radius.")
        if abs(offset_x) + abs(offset_y) < 0.20:
            break
    cx, cy = center
    if not (-0.10 * work_width <= cx <= 1.10 * work_width and -0.10 * work_height <= cy <= 1.10 * work_height):
        raise RuntimeError("Estimated wafer centre is outside the image.")

    if roi_center is not None:
        effective_search_center_angle_deg = float(
            math.degrees(math.atan2(roi_center[1] - cy, roi_center[0] - cx))
            % 360.0
        )
        search_distance = _angle_distance_deg(
            angles_deg, effective_search_center_angle_deg
        )
        search_mask = search_distance <= float(search_half_width_deg)

    inward_range = max(8.0, radius * (1.0 - float(radial_inner_ratio)))
    outward_range = max(5.0, radius * 0.018)
    boundary, support = _track_outer_edge(
        edge,
        center,
        radius,
        angles,
        inward_px=inward_range,
        outward_px=outward_range,
    )

    raw_depth = float(radius) - boundary
    supported_fit = fit_mask & (support >= np.percentile(support[fit_mask], 30.0))
    baseline_shift = float(np.median(raw_depth[supported_fit]))
    deficit = raw_depth - baseline_shift
    samples_per_degree = angle_samples / 360.0
    deficit = _circular_gaussian(
        deficit, max(3, int(round(0.30 * samples_per_degree)) | 1)
    )
    fit_residual = deficit[supported_fit]
    fit_residual_median = float(np.median(fit_residual))
    radial_noise = float(
        1.4826 * np.median(np.abs(fit_residual - fit_residual_median))
    )

    # ``baseline_window_deg`` remains accepted so old copy/paste calls do not
    # break. Geometry fitting outside the search sector replaces that baseline.
    _ = baseline_window_deg
    depth_limit = (
        float(min_notch_depth_px) * scale
        if min_notch_depth_px is not None
        else max(1.25, float(radius) * float(min_notch_depth_ratio))
    )
    candidate_threshold = max(
        0.50, depth_limit * 0.40, 2.5 * radial_noise
    )
    # Do not reject a notch because its edge is weak: low contrast is exactly
    # the difficult production case. Edge support contributes to confidence,
    # while geometry (depth/width/area) decides whether the depression exists.
    active = search_mask & (deficit >= candidate_threshold)
    # The notch edge can be interrupted by glare, die streets, or texture.
    # Join only short angular gaps; this preserves one physical depression
    # without requiring every ray to contain a strong gradient.
    bridge_kernel = max(3, int(round(0.80 * samples_per_degree)) | 1)
    half_bridge = bridge_kernel // 2
    extended_active = np.concatenate(
        (active[-half_bridge:], active, active[:half_bridge])
    ).astype(np.uint8).reshape(1, -1)
    extended_active = cv2.morphologyEx(
        extended_active,
        cv2.MORPH_CLOSE,
        np.ones((1, bridge_kernel), np.uint8),
    ).reshape(-1)
    active = extended_active[half_bridge:half_bridge + angle_samples].astype(bool)
    active &= search_mask
    groups = _circular_candidate_groups(active)
    degree_step = 360.0 / angle_samples
    candidates = []
    for indices in groups:
        width_deg = float(len(indices) * degree_step)
        if width_deg > 2.0 * float(search_half_width_deg):
            continue
        values = np.maximum(deficit[indices], 0.0)
        peak = float(values.max())
        area = float(values.sum() * degree_step)
        candidate_support = float(np.mean(support[indices]))
        score = area * math.sqrt(max(peak, 0.0)) * max(candidate_support, 0.02)
        candidates.append((score, peak, area, width_deg, candidate_support, indices))
    if candidates:
        _, peak_depth, candidate_area, notch_width_deg, candidate_support, candidate_indices = max(
            candidates, key=lambda item: item[0]
        )
        peak_index = int(candidate_indices[np.argmax(deficit[candidate_indices])])
    else:
        search_indices = np.flatnonzero(search_mask)
        peak_index = int(search_indices[np.argmax(deficit[search_indices])])
        peak_depth = float(deficit[peak_index])
        candidate_area = 0.0
        notch_width_deg = degree_step
        candidate_support = float(support[peak_index])
        candidate_indices = np.asarray((peak_index,), dtype=np.int64)

    minimum_width = degree_step
    strong_notch = bool(
        peak_depth >= max(depth_limit, 0.70, 4.5 * radial_noise)
        and notch_width_deg >= minimum_width
    )
    wide_shallow_notch = bool(
        peak_depth >= max(depth_limit * 0.45, 0.60, 3.0 * radial_noise)
        and notch_width_deg >= float(min_wide_notch_deg)
        and candidate_area >= depth_limit * max(1.5, float(min_wide_notch_deg))
    )
    found = bool(strong_notch or wide_shallow_notch)

    # The requested reference is the angular midpoint of the separated notch
    # region, not the depth-weighted apex. Unwrap group indices so a notch
    # crossing 0/360 degrees is handled correctly.
    unwrapped = np.unwrap(
        candidate_indices.astype(np.float64) * 2.0 * math.pi / angle_samples
    )
    notch_center_index = float(np.mean(unwrapped) * angle_samples / (2.0 * math.pi))
    notch_angle_rad = float(
        (notch_center_index % angle_samples) * 2.0 * math.pi / angle_samples
    )
    notch_angle_deg = float(math.degrees(notch_angle_rad) % 360.0)
    deepest_angle = float(angles[peak_index])
    notch_radius = float(boundary[peak_index])
    notch_deepest_point = (
        float(cx + math.cos(deepest_angle) * notch_radius),
        float(cy + math.sin(deepest_angle) * notch_radius),
    )
    # This is the user-confirmed red point: the notch centre direction at the
    # fitted wafer outer circle, i.e. where the circle would be without a cut.
    notch_point = (
        float(cx + math.cos(notch_angle_rad) * radius),
        float(cy + math.sin(notch_angle_rad) * radius),
    )

    notch_width_px = float(
        2.0 * radius * math.sin(math.radians(notch_width_deg) / 2.0)
    )
    snr = peak_depth / max(0.25, radial_noise)
    depth_score = (peak_depth - depth_limit) / max(depth_limit * 2.0, 1.0)
    area_score = candidate_area / max(depth_limit * 6.0, 1.0)
    confidence = float(np.clip(
        0.35 * min(1.0, snr / 8.0)
        + 0.25 * np.clip(depth_score, 0.0, 1.0)
        + 0.20 * np.clip(area_score, 0.0, 1.0)
        + 0.20 * np.clip(candidate_support, 0.0, 1.0),
        0.0,
        1.0,
    ))
    detection_method = "geometry_edge_bottom_sector"
    local_arc: Optional[Tuple[_Point, ...]] = None
    semicircle_candidate: Optional[_LocalSemicircleCandidate] = None
    fallback_attempted = False
    fallback_used = False
    fallback_reason = ""
    fallback_candidate: Optional[_RimIntrusionCandidate] = None
    if roi_center is not None and roi_half_size is not None:
        if background_geometry is not None:
            semicircle_candidate = _fit_semicircle_from_background_boundary(
                background_geometry,
                roi_center,
                roi_half_size,
                semicircle_radius_range,
            )
            detection_method = "roi_background_connected_notch_arc"
        else:
            semicircle_candidate = _detect_semicircle_in_roi(
                edge,
                center,
                radius,
                roi_center,
                roi_half_size,
                semicircle_radius_range,
            )
            detection_method = "geometry_edge_manual_roi_semicircle"
        if semicircle_candidate is None:
            found = False
            confidence = 0.0
            candidate_support = 0.0
            peak_depth = 0.0
            notch_width_deg = 0.0
            notch_width_px = 0.0
            candidate_indices = np.asarray((), dtype=np.int64)
        else:
            local_arc = semicircle_candidate.arc_points
            semicircle_center = np.asarray(
                semicircle_candidate.center, dtype=np.float64
            )
            wafer_center_array = np.asarray(center, dtype=np.float64)
            outward_vector = semicircle_center - wafer_center_array
            outward_length = float(np.linalg.norm(outward_vector))
            if outward_length <= 1e-6:
                found = False
                outward_unit = np.asarray((0.0, 1.0), dtype=np.float64)
            else:
                outward_unit = outward_vector / outward_length
            inward_unit = -outward_unit
            notch_half_width = float(
                semicircle_candidate.radius_x
                if semicircle_candidate.radius_x is not None
                else semicircle_candidate.radius
            )
            notch_height = float(
                semicircle_candidate.radius_y
                if semicircle_candidate.radius_y is not None
                else semicircle_candidate.radius
            )
            deepest = semicircle_center + inward_unit * float(
                notch_height
            )
            notch_angle_rad = float(
                math.atan2(float(outward_unit[1]), float(outward_unit[0]))
                % (2.0 * math.pi)
            )
            notch_angle_deg = float(math.degrees(notch_angle_rad) % 360.0)
            notch_deepest_point = (float(deepest[0]), float(deepest[1]))
            notch_point = (
                float(cx + math.cos(notch_angle_rad) * radius),
                float(cy + math.sin(notch_angle_rad) * radius),
            )
            peak_depth = max(
                0.0,
                float(radius)
                - float(np.linalg.norm(deepest - wafer_center_array)),
            )
            notch_width_px = float(2.0 * notch_half_width)
            notch_width_deg = float(math.degrees(
                2.0 * math.asin(
                    min(1.0, notch_half_width / max(radius, 1e-6))
                )
            ))
            candidate_support = float(semicircle_candidate.edge_support)
            found = bool(
                outward_length > 1e-6
                and semicircle_candidate.score >= float(notch_semicircle_min_score)
                and semicircle_candidate.arc_coverage >= 0.55
                and peak_depth >= max(0.60, depth_limit * 0.40)
            )
            confidence = (
                float(semicircle_candidate.score) if found else 0.0
            )
            candidate_indices = np.asarray((), dtype=np.int64)
    # Retry only the explicit detector rejection, not arbitrary exceptions.
    # Successful primary geometry and all its existing diagnostics stay intact.
    if not found and fallback_mode == "rim_intrusion" and background_geometry is not None:
        fallback_attempted = True
        fallback_candidate, fallback_reason = _detect_roi_rim_intrusion(
            background_geometry, center, radius, roi_center,
            search_half_width_deg=float(search_half_width_deg),
            radial_inner_ratio=float(radial_inner_ratio),
            min_depth=depth_limit,
            radius_range=semicircle_radius_range,
        )
        # A failed analytic arc is not an accepted fallback observation.
        semicircle_candidate = None
        local_arc = ()
        if fallback_candidate is not None:
            fallback_used = True
            found = True
            detection_method = "roi_background_rim_intrusion_fallback"
            notch_angle_deg = fallback_candidate.angle_deg
            notch_angle_rad = math.radians(notch_angle_deg)
            notch_point = (
                float(cx + math.cos(notch_angle_rad) * radius),
                float(cy + math.sin(notch_angle_rad) * radius),
            )
            notch_deepest_point = fallback_candidate.deepest_point
            peak_depth = fallback_candidate.depth
            notch_width_deg = fallback_candidate.width_deg
            notch_width_px = float(2.0 * radius * math.sin(math.radians(notch_width_deg) * 0.5))
            confidence = fallback_candidate.confidence
            candidate_support = 1.0  # All retained rays reach the connected exterior.
            local_arc = fallback_candidate.arc_points
            candidate_indices = np.asarray((), dtype=np.int64)
    if not found:
        confidence = 0.0
        if mode == "error":
            roi_message = ""
            if roi_center is not None:
                roi_message = (
                    f" roi_center=({roi_center[0] / scale:.1f},"
                    f"{roi_center[1] / scale:.1f}),"
                    f" semicircle_score={0.0 if semicircle_candidate is None else semicircle_candidate.score:.3f}."
                )
            raise RuntimeError(
                f"Wafer notch was not found: peak_depth={peak_depth / scale:.2f}px, "
                f"width={notch_width_deg:.2f}deg, required_depth={depth_limit / scale:.2f}px. "
                f"search={effective_search_center_angle_deg:.1f}+/-{float(search_half_width_deg):.1f}deg."
                f"{roi_message} fallback={fallback_reason or 'not_attempted'}. "
                "Use failure_mode='zero' to return angle 0, or correct the ROI/wafer hints."
            )
        notch_angle_deg = float(reference_angle_deg) % 360.0
        notch_angle_rad = math.radians(notch_angle_deg)
        notch_point = (
            float(cx + math.cos(notch_angle_rad) * radius),
            float(cy + math.sin(notch_angle_rad) * radius),
        )
        notch_deepest_point = notch_point
        candidate_indices = np.asarray((), dtype=np.int64)

    inv_scale = 1.0 / scale
    full_center = (float(cx * inv_scale), float(cy * inv_scale))
    full_radius = float(radius * inv_scale)
    full_notch_point = (
        float(notch_point[0] * inv_scale),
        float(notch_point[1] * inv_scale),
    )
    full_deepest_point = (
        float(notch_deepest_point[0] * inv_scale),
        float(notch_deepest_point[1] * inv_scale),
    )
    if local_arc is not None:
        arc = tuple(
            (float(point[0] * inv_scale), float(point[1] * inv_scale))
            for point in local_arc
        )
    else:
        arc = tuple(
            (
                float((cx + math.cos(angles[index]) * boundary[index]) * inv_scale),
                float((cy + math.sin(angles[index]) * boundary[index]) * inv_scale),
            )
            for index in candidate_indices[::max(1, len(candidate_indices) // 48)]
        )
    if background_geometry is not None:
        contour_full = np.rint(
            background_geometry.wafer_contour.astype(np.float64) * inv_scale
        ).astype(np.int32)
    else:
        contour_stride = max(1, angle_samples // 1440)
        contour_indices = np.arange(0, angle_samples, contour_stride, dtype=np.int64)
        contour_points = np.column_stack((
            cx + np.cos(angles[contour_indices]) * boundary[contour_indices],
            cy + np.sin(angles[contour_indices]) * boundary[contour_indices],
        ))
        contour_full = np.rint(contour_points * inv_scale).astype(np.int32).reshape(-1, 1, 2)
    if background_geometry is None:
        palette_bgr: Tuple[Tuple[int, int, int], ...] = ()
        background_threshold = 0.0
    else:
        palette_lab_u8 = np.clip(
            np.rint(background_geometry.palette_lab), 0, 255
        ).astype(np.uint8).reshape(1, -1, 3)
        converted_palette = cv2.cvtColor(palette_lab_u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)
        palette_bgr = tuple(
            tuple(int(value) for value in colour) for colour in converted_palette
        )
        background_threshold = float(background_geometry.distance_threshold_lab)
    return _NotchAngleResult(
        found=found,
        wafer_center_px=full_center,
        wafer_radius_px=full_radius,
        notch_point_px=full_notch_point,
        notch_deepest_point_px=full_deepest_point,
        notch_angle_deg=notch_angle_deg,
        reference_angle_deg=float(reference_angle_deg),
        correction_angle_deg=_normalise_angle(
            notch_angle_deg - float(reference_angle_deg)
        ),
        notch_depth_px=float(peak_depth * inv_scale),
        notch_width_deg=notch_width_deg,
        notch_width_px=float(notch_width_px * inv_scale),
        confidence=confidence,
        radial_noise_px=float(radial_noise * inv_scale),
        candidate_arc_px=arc,
        wafer_contour_px=contour_full,
        segmentation_threshold=float(edge_normaliser),
        scale=scale,
        failure_mode=mode,
        detection_method=detection_method,
        search_center_angle_deg=effective_search_center_angle_deg,
        search_half_width_deg=float(search_half_width_deg),
        edge_support=float(candidate_support),
        circle_fit_residual_px=float(circle_fit_noise * inv_scale),
        roi_center_px=(
            None
            if roi_center is None
            else (float(roi_center[0] * inv_scale), float(roi_center[1] * inv_scale))
        ),
        roi_bounds_px=(
            (tuple(float(value * inv_scale) for value in background_geometry.roi_bounds)
             if fallback_attempted else None)
            if semicircle_candidate is None
            else tuple(float(value * inv_scale) for value in semicircle_candidate.roi_bounds)
        ),
        semicircle_center_px=(
            None
            if semicircle_candidate is None
            else (
                float(semicircle_candidate.center[0] * inv_scale),
                float(semicircle_candidate.center[1] * inv_scale),
            )
        ),
        semicircle_radius_px=(
            None
            if semicircle_candidate is None
            else float(semicircle_candidate.radius * inv_scale)
        ),
        semicircle_radius_x_px=(
            None
            if semicircle_candidate is None
            else float(
                (
                    semicircle_candidate.radius_x
                    if semicircle_candidate.radius_x is not None
                    else semicircle_candidate.radius
                )
                * inv_scale
            )
        ),
        semicircle_radius_y_px=(
            None
            if semicircle_candidate is None
            else float(
                (
                    semicircle_candidate.radius_y
                    if semicircle_candidate.radius_y is not None
                    else semicircle_candidate.radius
                )
                * inv_scale
            )
        ),
        semicircle_shape=(
            "none" if semicircle_candidate is None else semicircle_candidate.shape
        ),
        semicircle_score=(
            0.0 if semicircle_candidate is None else float(semicircle_candidate.score)
        ),
        semicircle_fit_residual_px=(
            0.0
            if semicircle_candidate is None
            else float(semicircle_candidate.fit_residual * inv_scale)
        ),
        background_segmentation_used=background_geometry is not None,
        background_palette_bgr=palette_bgr,
        background_distance_threshold_lab=background_threshold,
        fallback_attempted=fallback_attempted,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        notch_shoulder_points_px=(
            None if fallback_candidate is None else tuple(
                (float(point[0] * inv_scale), float(point[1] * inv_scale))
                for point in fallback_candidate.shoulder_points
            )
        ),
        fallback_angle_stability_deg=(
            0.0 if fallback_candidate is None else fallback_candidate.angle_stability_deg
        ),
    )


__all__ = ["angle"]


def angle(src) -> np.ndarray:
    """Return a new image with its wafer notch aligned to the bottom.

    ``src`` is a uint8 ndarray (H,W), (H,W,3) BGR, or (H,W,4) BGRA, or a
    str/Path image filename. Unicode paths are decoded without cv2.imread.
    Shape, channel count, and dtype are preserved. The input is never written.

    Invalid arrays or undecodable files raise ValueError/TypeError. Missing
    files raise FileNotFoundError; unreliable wafer/notch detection raises
    RuntimeError. There is no silent fallback to an uncorrected image.
    """
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

    # Detection uses BGR; resampling uses the original channel representation.
    if image.ndim == 2:
        detection_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        detection_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        detection_image = image
    result = _detect_wafer_notch(detection_image)
    matrix = cv2.getRotationMatrix2D(result.wafer_center_px, result.correction_angle_deg, 1.0)
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
