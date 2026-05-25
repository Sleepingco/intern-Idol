# Chin Strap AR (실시간 얼굴 추적 기반 3D 착용형 AR)

실시간 웹캠 영상 위에 **3D 갓(모자) + 스트랩(끈) + 2D 메이크업 이펙트**를 합성하는 컴퓨터비전/그래픽스 프로젝트입니다.  
MediaPipe/ONNX 기반 얼굴·포즈·세그멘테이션 결과를 OpenGL 렌더링 파이프라인과 결합해, 사람 분리(오클루전)까지 포함한 AR 경험을 구현했습니다.

---

## 1) 프로젝트 한 줄 소개

> "한 사람(오너)만 안정적으로 추적해 3D 소품과 2D 스타일 효과를 자연스럽게 합성하는 실시간 AR 엔진"

---

## 2) 주요 구현 포인트

- **실시간 얼굴 랜드마크 기반 3D 정합**
  - 얼굴 랜드마크/회전 정보를 활용해 모자(GLB/OBJ)와 끈을 3D 공간에 배치.
  - 카메라 내재 파라미터 기반 투영 행렬 지원으로 정합 안정성 강화.

- **오클루전(가림) 처리로 현실감 향상**
  - 턱 마스크/사람 마스크를 이용해 3D 오브젝트가 얼굴·신체 뒤로 자연스럽게 가려지도록 처리.
  - 배경 분리 및 오너 중심 게이팅(connected component, pose ROI)을 통해 다중 인물 상황에서 대상 고정.

- **2D 뷰티/스타일 이펙트 파이프라인**
  - 스킨톤, 스모키아이, 홍채/동공, 립컬러 등 복합 이펙트 지원.
  - ONNX Runtime provider 자동 선택(CUDA 우선, 실패 시 CPU fallback).

- **물리 기반 스트랩(rope) 시뮬레이션**
  - 중력/감쇠/제약 반복 기반으로 끈의 움직임을 자연스럽게 표현.

- **실전형 실행 환경 고려**
  - 전체화면/창 모드 전환, 카메라 종횡비 대응 뷰포트 계산, 배경 이미지 fit 모드 지원.

---

## 3) 기술 스택

- **Language**: Python
- **Vision/ML**: MediaPipe, ONNX Runtime
- **Graphics**: PyOpenGL, Pygame
- **Image Processing**: OpenCV, NumPy, imageio
- **3D Assets**: OBJ/GLB, texture maps

---

## 4) 시스템 구성

```text
[Webcam Frame]
   ├─ Face detection/landmark (MediaPipe)
   ├─ Pose detection (MediaPipe Pose)
   ├─ Segmentation (ONNX Runtime)
   │
   ├─ Owner tracking + mask gating
   ├─ 2D effects (skin/eye/lip/smoky...)
   └─ 3D transform + physics (hat/strap)
            ↓
      OpenGL compositing
            ↓
        Final AR Output
```

---

## 5) 실행 방법

### 5-1. 의존성 설치

```bash
pip install -r requirements.txt
```

> GPU 환경에서는 `onnxruntime-gpu` 동작 여부를 확인하세요.

### 5-2. 실행

```bash
python main.py
```

### 5-3. 기본 조작

- `ESC`: 종료

---

## 6) 프로젝트 구조

```text
.
├─ main.py                 # 메인 루프/렌더링 오케스트레이션
├─ config.py               # 카메라·렌더링·이펙트·물리 파라미터
├─ face_processing.py      # 얼굴 검출/랜드마크/마스크 생성
├─ pose_processing.py      # 포즈 검출 및 오너 ROI 마스크
├─ face_effects.py         # ONNX 세그 + 2D 스타일 이펙트
├─ rendering.py            # OpenGL 렌더링 유틸
├─ physics.py              # 스트랩 물리 계산
├─ input_handler.py        # 키 입력 처리
├─ app/                    # 카메라/디스플레이/변환 보조 모듈
├─ sticker3d/              # 3D 스티커 렌더러
├─ asset/, obj/, models/   # 모델/텍스처/온디스크 리소스
└─ requirements.txt
```

---

## 7) 특징

- **CV + Graphics 융합 역량**: 랜드마크 기반 pose 정합과 OpenGL 렌더링의 end-to-end 통합.
- **실시간 최적화/안정화 경험**: 마스크 스무딩, 게이팅, fallback 경로 설계.
- **사용자 경험 고려**: 다중 인물 환경에서 오너 고정, 자연스러운 가림 처리.
- **실무형 디버깅 경험**: GPU provider 우선 사용, 환경 의존성 대응 로직 포함.

---

## 8) 향후 개선 아이디어

- 추적 안정화를 위한 temporal filter(예: Kalman/OneEuro) 고도화.
- 리소스 로딩/렌더 패스 분리를 통한 모듈성 강화.
- UI 오버레이(실시간 파라미터 튜닝) 및 preset 시스템 추가.
- 배포를 위한 환경 자동화(예: Docker, 실행 스크립트 정리).

---

## 9) 데모/결과물 섹션 (추가 권장)

포트폴리오 제출 시 아래를 함께 넣으면 전달력이 크게 좋아집니다.

- 실행 GIF 또는 20~40초 데모 영상
- "원본 영상 vs 적용 결과" 비교 이미지
- 기술 챌린지 3가지와 해결 방식(문제-가설-실험-결론)
