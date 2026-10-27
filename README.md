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

## 입력과 반환

| 구분 | 내용 |
|---|---|
| 입력 | `uint8` NumPy 이미지 또는 `str` / `pathlib.Path` 파일 경로 |
| 지원 이미지 | 회색조 `(H, W)`, BGR `(H, W, 3)`, BGRA `(H, W, 4)` |
| 반환 | 보정된 `numpy.ndarray` 한 개 |
| 보존 | 이미지 크기, 채널 수, `uint8` 자료형 |
| 원본 | 입력 배열을 변경하지 않고 새 배열을 반환 |
| 회전 | 검출된 웨이퍼 중심 기준, OpenCV cubic 보간 |
| 빈 영역 | 회색조/BGR은 검정, BGRA는 투명한 검정 |

같은 크기의 캔버스를 사용하므로 회전한 이미지의 모서리 부분은 잘릴 수 있습니다.

## 보정 기준

기존 V7의 **웨이퍼 외곽 노치 검출 방식**을 분리했습니다. 내부 격자선을 보고 수평을 맞추는 방식이 아닙니다.

1. 이미지의 색상 경계에서 웨이퍼 외곽과 중심을 구합니다.
2. 아래쪽 방향을 중심으로 ±45° 범위에서 노치를 찾습니다.
3. 노치가 아래쪽을 향하도록 이미지를 회전합니다.
4. 회전한 이미지 배열만 반환합니다.

이미지 좌표의 오른쪽이 0°, 아래쪽이 90°입니다. 보정은 노치가 대체로 하단에 보이는 전체 웨이퍼 이미지를 전제로 합니다. 노치가 없는 중심 clip이나 임의 360° 방향의 입력에 대한 자동 정렬을 보장하지 않습니다.

`refine_mode`, `refine_radius` 등 교차점 중심 보정 옵션은 이 함수와 관계없습니다. 공개 함수의 입력은 `src` 하나입니다.

## 입력 또는 검출에 문제가 있을 때

| 상황 | 예외 |
|---|---|
| 지원하지 않는 입력 종류·자료형 | `TypeError` |
| 빈 이미지·지원하지 않는 채널 형태·디코드 불가 파일 | `ValueError` |
| 파일 경로가 없음 | `FileNotFoundError` |
| 웨이퍼 외곽 또는 신뢰할 수 있는 노치를 찾지 못함 | `RuntimeError` |

각도를 구하지 못했을 때 보정되지 않은 원본을 성공 결과처럼 반환하지 않습니다.

## 검증

```bash
python -m unittest discover -s tests -v
```

테스트는 알려진 각도로 회전한 합성 웨이퍼, 회색조·BGR·BGRA, 입력 불변성, 한글 경로, 실패 입력과 독립 import를 확인합니다. 실제 장비에서 발생한 실패 이미지에 대한 검증은 포함하지 않습니다.

## 출처

노치 검출 코드는 [NS-DDC/Wafer_Via의 `wafer_notch_angle.py`](https://github.com/NS-DDC/Wafer_Via/blob/5adf92eb0d91fb5bbb1e09517d9688d68c2e7c18/codex/wafer_notch_angle.py)에서 필요한 의존 함수만 추출했습니다. 공개 API를 `angle(src)`로 제한하고 검출용 색 변환과 출력용 회전을 분리해 원본 채널 구성을 유지합니다.
