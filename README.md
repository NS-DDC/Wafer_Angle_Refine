# Wafer Angle Refine

웨이퍼 이미지에서 **노치를 기준으로 각도를 보정한 이미지 한 장**만 반환합니다.

```python
from angle import angle

Angle_refine = angle(src)
```

YOLO 모델, 검출 좌표, 중심 clip, 격자 생성, die 조회가 필요하지 않습니다. 오버레이·좌표·행렬·진단 객체를 반환하지 않고, 함수를 호출하는 것만으로 파일을 저장하거나 창을 띄우지 않습니다.

## 설치

Python 3.10 이상에서 사용합니다.

```bash
python -m pip install -r requirements.txt
```

이미 NumPy와 OpenCV가 설치된 프로젝트라면 `angle.py` 파일만 복사해 사용할 수 있습니다. 기존 `Wafer_Via` 저장소를 설치하거나 함께 복사할 필요가 없습니다.

## 사용

OpenCV 이미지 배열을 넣습니다.

```python
import cv2
from angle import angle

src = cv2.imread("wafer.png", cv2.IMREAD_UNCHANGED)
Angle_refine = angle(src)
cv2.imwrite("wafer_aligned.png", Angle_refine)
```

파일 경로도 직접 넣을 수 있습니다. 한글 경로를 지원합니다.

```python
from angle import angle

Angle_refine = angle("샘플/웨이퍼.png")
```

### 기존 V7 노치 설정 전달

`angle(src)` 호출은 그대로 사용할 수 있습니다. 추가 옵션은 모두 **이름을 지정하는 keyword-only 인자**이며, 기존 V7 `wafer_via_notch.py`의 `build_die_map_from_yolo()`에서 쓰던 검출용 `notch_*` 이름과 기본값을 유지합니다.

아래 변수에는 기존 프로젝트에서 사용하는 ROI·웨이퍼 중심·반경 값을 넣습니다. 예시용 임의 숫자를 실제 이미지 설정으로 가정하지 않습니다. 힌트나 노치 반경 범위를 사용하지 않았다면 해당 항목을 생략하면 됩니다.

```python
from angle import angle

# roi_x, roi_y 등은 기존 프로젝트에서 얻은 원본 이미지 좌표/설정값입니다.
notch_options = {
    "notch_roi_center_px": (roi_x, roi_y),
    "notch_roi_half_size_px": (roi_half_width, roi_half_height),
    "notch_wafer_center_hint_px": (wafer_cx, wafer_cy),
    "notch_wafer_radius_hint_px": wafer_radius_px,
    "notch_semicircle_radius_range_px": (notch_radius_min_px, notch_radius_max_px),
    "notch_failure_mode": "error",
}

Angle_refine = angle(src, **notch_options)
```

이미 검출용 노치 옵션만 담은 `notch_options` 사전이 있다면 마지막 호출에 그대로 전달할 수 있습니다. YOLO·격자·중심 교차점 보정 옵션, `return_notch_visuals`, `notch_visual_max_dimension`, `notch_zoom_size_px`, `notch_zoom_scale`, 이미지 반환 여부 옵션은 이 함수에 전달하지 않습니다. 반환은 옵션과 관계없이 이미지 한 장입니다.

### 옵션과 기본값

`*_px` 위치·거리 값은 모두 **축소 전 입력 원본 이미지의 pixel** 기준입니다. 노치 ROI의 `(x, y)`도 전체 원본 이미지의 좌표이며, crop 내부 좌표나 `notch_max_dimension`으로 축소된 영상의 좌표가 아닙니다. 함수 내부에서 필요한 크기로 변환합니다. 각도는 오른쪽 0°, 아래쪽 90°인 이미지 좌표 기준입니다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `notch_reference_angle_deg` | `90.0` | 보정 후 노치가 향할 목표 각도. 기본값은 아래쪽 |
| `notch_max_dimension` | `3072` | 검출용 작업 영상의 최대 변 길이(px). 출력 이미지 크기는 바꾸지 않음 |
| `notch_angle_samples` | `3600` | 원주 방향 샘플 수. 내부에서 최소 720으로 처리 |
| `notch_baseline_window_deg` | `10.0` | **이전 호출과의 호환용으로만 유지. 현재 알고리즘에서 사용하지 않아 값을 바꿔도 효과 없음** |
| `notch_min_depth_px` | `None` | 노치 최소 깊이의 원본 px 값. 지정하면 아래 비율 대신 사용 |
| `notch_min_depth_ratio` | `0.001` | 최소 깊이를 웨이퍼 반경 비율로 정함. `notch_min_depth_px=None`일 때 사용하며 작업 영상에서 최소 1.25px 적용 |
| `notch_min_wide_deg` | `2.0` | 넓고 얕은 노치 판정에 사용하는 최소 각도 폭 |
| `notch_search_center_angle_deg` | `90.0` | ROI 미지정 시 노치 탐색 중심 방향 |
| `notch_search_half_width_deg` | `45.0` | 탐색 중심 양쪽의 반폭(°), 허용 범위 2~120 |
| `notch_wafer_center_hint_px` | `None` | 웨이퍼 중심 `(x, y)` 힌트. ROI 중심과 다른 값이며 초기 외곽 추정을 도움 |
| `notch_wafer_radius_hint_px` | `None` | 웨이퍼 반경 힌트(px). 노치 자체의 반경과 다름 |
| `notch_roi_center_px` | `None` | 예상 노치 ROI 중심 `(x, y)`. 지정하면 해당 ROI 방향으로 탐색 중심을 계산 |
| `notch_roi_half_size_px` | `600.0` | ROI 반크기. 숫자 하나 또는 `(half_width, half_height)`, 각 값은 양수. 600이면 요청 영역은 중심 양쪽 600px |
| `notch_semicircle_radius_range_px` | `None` | ROI의 노치 형상 반경 범위 `(min, max)`. `0 < min < max`. `None`이면 자동 설정 |
| `notch_semicircle_min_score` | `0.55` | ROI 노치 형상 점수 하한, 0~1 |
| `notch_use_roi_background` | `True` | ROI 바깥쪽에서 배경색을 학습하고 경계와 연결된 배경으로 노치 형상을 검사. ROI를 지정했을 때 적용 |
| `notch_background_palette_size` | `3` | ROI 배경 대표색 수, 1~8 |
| `notch_background_outer_band_fraction` | `0.28` | ROI 바깥쪽 배경 학습 구간의 비율, 0.10~0.60 |
| `notch_background_distance_threshold_lab` | `None` | 배경색으로 분류할 Lab 거리의 허용 상한을 직접 지정. `None`이면 자동, 지정 시 양수 |
| `notch_background_noise_margin_lab` | `4.0` | 자동 배경색 거리 임계값에 추가하는 노이즈 여유, 0 이상 |
| `notch_background_morph_px` | `24.0` | ROI 배경 마스크의 형태학적 처리 크기, 원본 px 기준 양수. 내부에서 축소율·kernel 제한 적용 |
| `notch_fallback_mode` | `"rim_intrusion"` | ROI 형상 검출이 거부되면 같은 배경 연결성과 외곽 원에서 안쪽 침범을 재검사. `"none"`이면 비활성화 |
| `notch_failure_mode` | `"error"` | 신뢰할 노치 미검출 시 `RuntimeError`. `"zero"`는 아래 실패 정책에 따라 0° 이미지 반환 |
| `alignment_interpolation` | `cv2.INTER_CUBIC` | 출력 이미지 회전 보간법. 기존 코드의 OpenCV 보간 상수를 전달 |
| `alignment_border_value` | `(0, 0, 0)` | 회전 후 입력이 없는 영역에 채울 값. 기본값은 검정이며 BGRA의 기본 alpha는 0 |

ROI 관련 형상·배경 옵션은 `notch_roi_center_px`를 지정했을 때 의미가 있습니다. `notch_background_*`는 ROI 배경 학습 경로를 켰을 때 사용합니다. `notch_fallback_mode="rim_intrusion"` 역시 배경 ROI가 있어야 적용 가능하며, 이미 성공한 1차 검출 결과를 바꾸거나 일반 예외를 숨기지 않습니다. 중심·반경 힌트는 항상 고정된 정답으로 취급되는 값이 아니며, 경로에 따라 외곽 fitting으로 조정될 수 있습니다.

보간법이나 빈 영역 색도 이름으로 지정할 수 있습니다.

```python
import cv2

Angle_refine = angle(
    src,
    **notch_options,
    alignment_interpolation=cv2.INTER_LINEAR,
    alignment_border_value=(0, 0, 0),
)
```

`alignment_border_value`는 OpenCV에 그대로 전달합니다. 회색조에는 `0` 같은 정수, BGR에는 `(B, G, R)`, BGRA에는 `(B, G, R, A)`를 사용하십시오. 기본 `(0, 0, 0)`은 BGRA에서 생략된 alpha가 0으로 처리되어 투명한 검정이 됩니다. 정수 하나를 BGR 전체 채널에 자동 복제하는 별도 처리는 없습니다.

## 입력과 반환

| 구분 | 내용 |
|---|---|
| 입력 | `uint8` NumPy 이미지 또는 `str` / `pathlib.Path` 파일 경로 |
| 지원 이미지 | 회색조 `(H, W)`, BGR `(H, W, 3)`, BGRA `(H, W, 4)` |
| 반환 | 보정된 `numpy.ndarray` 한 개 |
| 보존 | 이미지 크기, 채널 수, `uint8` 자료형 |
| 원본 | 입력 배열을 변경하지 않고 새 배열을 반환 |
| 회전 | 검출된 웨이퍼 중심 기준. 기본 OpenCV cubic, `alignment_interpolation`으로 조절 |
| 빈 영역 | 기본 회색조/BGR은 검정, BGRA는 투명한 검정. `alignment_border_value`로 조절 |

같은 크기의 캔버스를 사용하므로 회전한 이미지의 모서리 부분은 잘릴 수 있습니다.

## 보정 기준

기존 V7의 **웨이퍼 외곽 노치 검출 방식**을 분리했습니다. 내부 격자선을 보고 수평을 맞추는 방식이 아닙니다.

1. 이미지의 색상 경계에서 웨이퍼 외곽과 중심을 구합니다.
2. 기본적으로 아래쪽 방향을 중심으로 ±45° 범위에서 노치를 찾습니다. 탐색 방향·폭 또는 ROI를 지정할 수 있습니다.
3. 노치가 목표 방향을 향하도록 이미지를 회전합니다. 기본 목표는 아래쪽입니다.
4. 회전한 이미지 배열만 반환합니다.

이미지 좌표의 오른쪽이 0°, 아래쪽이 90°입니다. 보정은 노치가 대체로 하단에 보이는 전체 웨이퍼 이미지를 전제로 합니다. 노치가 없는 중심 clip이나 임의 360° 방향의 입력에 대한 자동 정렬을 보장하지 않습니다.

`refine_mode`, `refine_radius` 등 교차점 중심 보정 옵션은 이 함수와 관계없습니다. 필수 입력은 `src` 하나이며, 노치 검출용 23개와 출력 회전용 2개의 선택 인자를 이름으로 전달할 수 있습니다.

## 입력 또는 검출에 문제가 있을 때

| 상황 | 예외 |
|---|---|
| 지원하지 않는 입력 종류·자료형 | `TypeError` |
| 빈 이미지·지원하지 않는 채널 형태·디코드 불가 파일 | `ValueError` |
| 파일 경로가 없음 | `FileNotFoundError` |
| 잘못된 노치 옵션 범위·지원하지 않는 설정값 | `ValueError` 등 해당 입력 검증 예외 |
| 웨이퍼 외곽을 구하지 못함 | `RuntimeError` (`notch_failure_mode="zero"`도 해결하지 않음) |
| 외곽은 구했지만 신뢰할 수 있는 노치를 찾지 못함 | 기본 `"error"`이면 `RuntimeError`; `"zero"`이면 회전하지 않은 새 이미지 |

기본 `notch_failure_mode="error"`는 노치 미검출을 예외로 알립니다. 미검출을 0°로 처리하려는 경우에만 `angle(src, notch_failure_mode="zero")`를 지정합니다. 이 경우 **웨이퍼 외곽을 정상적으로 구한 뒤 노치만 미검출되었을 때** 입력과 같은 픽셀을 가진 새 이미지를 반환합니다. 파일 오류, 잘못된 옵션, 외곽 추정 실패까지 포괄적으로 무시하는 설정은 아닙니다.

이 API는 이미지 한 장만 반환하므로 `"zero"` 결과 자체에는 노치 검출 성공 여부가 들어 있지 않습니다. 실제 0° 보정과 미검출로 인한 0°를 구분해야 한다면 기본 `"error"` 정책을 사용하십시오.

## 검증

```bash
python -m unittest discover -s tests -v
```

테스트는 알려진 각도로 회전한 합성 웨이퍼, 회색조·BGR·BGRA, 입력 불변성, 한글 경로, 실패 입력과 독립 import를 확인합니다. 실제 장비에서 발생한 실패 이미지에 대한 검증은 포함하지 않습니다.

## 출처

노치 검출 코드는 [NS-DDC/Wafer_Via의 `wafer_notch_angle.py`](https://github.com/NS-DDC/Wafer_Via/blob/5adf92eb0d91fb5bbb1e09517d9688d68c2e7c18/codex/wafer_notch_angle.py)에서 필요한 의존 함수만 추출했습니다. 공개 API는 이미지 한 장을 반환하는 `angle()`이며, `angle(src)` 기본 호출과 기존 V7 노치 설정을 받는 keyword-only 옵션을 지원합니다. 검출용 색 변환과 출력용 회전을 분리해 원본 채널 구성을 유지합니다.
