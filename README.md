# Wafer Angle Refine

웨이퍼 **내부 격자선을 수평·수직으로 맞춘 이미지 한 장**을 반환합니다.

~~~python
from angle import angle

Angle_refine = angle(src)
~~~

**angle.py 하나를 통째로 복사하면 됩니다.** 필요한 패키지는 NumPy와 OpenCV입니다. YOLO 모델이나 검출 좌표 없이 입력 이미지에서 격자 기울기를 구합니다. 함수는 파일을 저장하거나 창을 띄우지 않습니다.

기존 노치 기준 버전과 조절 파라미터는 [notch-reference 브랜치](https://github.com/NS-DDC/Wafer_Angle_Refine/tree/notch-reference)에 보존했습니다. main의 angle()은 내부 격자를 기준으로 합니다. 따라서 노치의 최종 위치는 아래쪽으로 고정되지 않습니다.

## 설치와 사용

Python 3.10 이상에서 사용합니다.

~~~bash
python -m pip install -r requirements.txt
~~~

~~~python
import cv2
from angle import angle

src = cv2.imread("wafer.png", cv2.IMREAD_UNCHANGED)
Angle_refine = angle(src)
cv2.imwrite("wafer_aligned.png", Angle_refine)
~~~

한글 파일 경로도 직접 넣을 수 있습니다.

~~~python
Angle_refine = angle("샘플/웨이퍼.png")
~~~

- 입력: uint8 회색조 (H, W), BGR (H, W, 3), BGRA (H, W, 4) 배열 또는 이미지 경로.
- 출력: 입력과 같은 크기·채널·dtype의 새 NumPy 배열. 입력 배열은 변경하지 않습니다.
- 회전 중심: 이미지 중앙. 캔버스 크기를 유지하므로 가장자리 일부가 잘리고 빈 영역이 생길 수 있습니다.
- 격자 근거가 부족하면 RuntimeError가 발생합니다. 호출자가 실패 시 원본을 사용할지 결정할 수 있습니다.

~~~python
try:
    Angle_refine = angle(src)
except RuntimeError as exc:
    print(f"격자 각도 보정 보류: {exc}")
    Angle_refine = src.copy()
~~~

## 동작

~~~mermaid
flowchart LR
    A[입력 이미지] --> B[중앙 영역 선택]
    B --> C[Lab 색상·밝기 경계]
    C --> D[회전각별 가로·세로 투영]
    C --> E[FFT 방향 분석]
    D --> F{두 축의 격자 근거 확인}
    E --> F
    F -->|확실함| G[원본을 이미지 중앙 기준으로 회전]
    F -->|불확실함| H[RuntimeError]
    G --> I[보정 이미지 한 장 반환]
~~~

Lab의 밝기와 두 색상 축에서 경계 에너지를 계산합니다. 흰 격자선과 초록 내부 패턴처럼 색상이 다른 경계도 분석 대상입니다. 후보 각도로 분석 이미지를 돌려 행·열 방향으로 합쳤을 때, 격자가 축에 맞으면 반복 피크가 선명해지는 성질을 이용합니다. 가로·세로 각각의 최적 방향과 반복 신호도 확인합니다.

FFT에서는 반복 주기가 강한 주파수 링의 방향 집중도가 충분할 때 해당 방향을 사용하고, 그렇지 않으면 넓은 주파수 대역의 추정값을 사용합니다. 내부 회로 무늬가 만드는 다른 방향의 영향을 줄이기 위한 검사입니다. 분석 중에는 여러 후보를 회전해 보지만, 최종 출력은 누적 각도로 원본을 한 번만 회전해 만듭니다.

이 작업은 **격자 전체의 기울기 보정**입니다. refine_cross_point처럼 개별 십자가 중심을 찾거나 die 박스를 생성하지 않습니다. 사각 격자는 90도마다 같은 축 정렬을 만들므로 가장 가까운 수평·수직 방향으로 맞춥니다. 격자만으로 위·아래 또는 90도 회전 여부를 판별하지는 않습니다.

## 파라미터

아래 파라미터는 모두 angle(src, 이름=값)으로 조절합니다. src 뒤의 옵션은 키워드로 전달합니다.

| 파라미터 | 기본값 | 용도 |
| --- | --- | --- |
| roi_ratio | 0.55 | 이미지의 짧은 변 대비 중앙 분석 정사각형의 한 변 비율. 출력 영역을 자르는 옵션은 아닙니다. |
| search_deg | 6.0 | 최초 탐색 범위: ±6도. |
| coarse_step | 0.15 | 큰 간격 탐색의 각도 간격(도). 작게 하면 계산량이 늘어납니다. |
| fine_step | 0.02 | 최적 후보 주변의 정밀 탐색 간격(도). |
| max_dim | 1100 | 분석용 영상의 최대 변 크기(px). 출력 해상도는 그대로입니다. |
| fft_max_dim | 1024 | FFT 분석용 최대 변 크기(px). |
| agree_tol_deg | 0.50 | 투영·FFT 및 가로·세로 각도의 합의 허용 차이(도). |
| full_scan_deg | 44.0 | 확대 탐색과 최종 보정각의 최대 절댓값(도). search_deg 이상, 45 이하로 지정합니다. |
| max_iter | 3 | 잔여 기울기를 확인·보정하는 최대 반복 횟수. |
| min_angle_deg | 0.01 | 이보다 작은 잔여각은 추가 보정을 멈춥니다. |
| alignment_interpolation | cv2.INTER_CUBIC | 최종 이미지 회전 보간법. |
| alignment_border_value | (0, 0, 0) | 회전으로 비는 영역의 색. BGRA에서는 4개 값을 지정할 수 있습니다. |

예를 들어 중앙 영역에 격자선이 너무 적으면 roi_ratio를 키우고, 얇은 선이 축소 과정에서 사라지면 max_dim을 키워볼 수 있습니다. ROI에 외곽 링이나 글자가 많이 들어오면 오히려 줄이는 편이 유리합니다.

~~~python
Angle_refine = angle(
    src,
    roi_ratio=0.70,
    search_deg=10.0,
    full_scan_deg=44.0,
    max_dim=1400,
    alignment_interpolation=cv2.INTER_CUBIC,
    alignment_border_value=(0, 0, 0),
)
~~~

agree_tol_deg를 크게 하면 불일치를 더 쉽게 받아들이므로, 보정이 실패할 때는 우선 격자선이 분석 영역에 충분히 들어오는지 확인하세요. 기본 탐색 한계는 ±44도이며 정확히 45도 부근까지 필요하면 full_scan_deg=45.0으로 지정할 수 있습니다. 파라미터를 조절해도 원근 왜곡이나 휘어진 격자를 직선 격자로 바꾸지는 않습니다.

## 검증

~~~bash
python -B -m unittest discover -s tests -v
~~~

2026-10-27 기준 자동 테스트 23개가 통과했습니다. 테스트 입력은 로컬에서 생성한 기하학적 격자입니다. 알려진 기울기와 독립적인 픽셀 선형회귀로 보정 결과를 검사합니다. 흰색/초록색, 동일 명도 색상, 회색조/BGRA, 양축 정렬, 부정확한 근거 거절, 탐색 범위, 45도 경계 및 angle.py만 복사한 독립 실행을 포함합니다.

추가로 기존 저장소의 웨이퍼 샘플 4장과 MIPS 샘플에 ±3도 회전을 넣은 2개 사례에서 출력·입력 보존을 확인했습니다. MIPS/Casio/Piper 및 주입 회전 사례의 독립 Canny/Hough 측정에서 최대 축 잔여 기울기는 약 0.013도 이내였습니다. Exposed 샘플은 내부 무늬 때문에 독립 Hough 측정이 일치하지 않아 수치 정확도 통과로 세지 않고, 보정 이미지의 격자선 정렬을 눈으로 확인한 결과로 구분합니다.

사용자 현장의 문제 이미지는 아직 제공되지 않았으므로 해당 이미지의 성공 여부는 별도 확인이 필요합니다.

## 코드 계보

색상 경계·투영·FFT의 기본 접근은 [NS-DDC/Wafer_color_v6의 wafer_color_v6_claude.py](https://github.com/NS-DDC/Wafer_color_v6/blob/9f3b7bf92f7dd2fc2ca111fbe33ae937fac6823b/wafer_color_v6_claude.py)에 있는 각도 추정 경로에서 가져왔습니다. FFT 주파수 링 선택은 V7의 wafer_via_die_render.py 접근도 참고했습니다. 이미지 한 장만 받도록 분리하면서 신호 부족·양축 반복성·각도 합의 검사와 입력/출력 처리를 추가했습니다. 원본 프로젝트를 설치하거나 함께 복사할 필요가 없습니다.
