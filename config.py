# -*- coding: utf-8 -*-
# config.py - 설정 파일
import numpy as np
import cv2

# -------------------- 카메라/투영 설정 --------------------
CAM_ID = 0
# HFOV_DEG = 80.0
NEAR_Z, FAR_Z = 0.1, 1000.0

# --- 앵커(눈썹/코 등 기준점) 설정 ---
ANCHOR_LM_IDX = 10  # 정중앙 부근(approx) – 초기 정렬에 유용한 랜드마크
ANCHOR_SMOOTH = 0.0  # 0.0(즉시 반영) ~ 0.95(매우 느리게 반영)

# -------------------- 모델/리소스 경로 --------------------
MODEL_PATH = "./obj/face_landmarker.task"
GLB_PATH = "./obj/bone2.glb"      # 끈/본(rope/bone) GLB
OBJ_PATH = "./obj/gat_M.obj"      # 갓 OBJ

# --- 좌표축(axes) 디버그 표시 ---
SHOW_AXES_HAT = False   # F6: 갓 축 표시
SHOW_AXES_STRAP = False # F7: 끈 축 표시
AXES_LEN = 2.0          # 축 길이(월드 좌표계 기준)
AXES_LINE_WIDTH = 3.0   # 축 선 두께
AXES_ALPHA = 1.0        # 축 투명도

# --- 시작 화면 모드 ---
START_FULLSCREEN = True  # 시작 시 전체화면

# --- 좌표축 Z 오프셋/스텝 ---
AXIS_Z_OFFSET = 0.0     # 축의 Z 위치 오프셋
AXIS_Z_STEP = 0.1       # Z 조정 스텝

# --- 오브젝트 스케일(크기) 조정 ---
HAT_SCALE_MULTIPLIER = 0.59   # 갓 스케일 계수
STRAP_SCALE_MULTIPLIER = 1.19 # 끈 스케일 계수
SCALE_STEP = 0.01             # 스케일 조정 스텝

# --- 오브젝트 위치 오프셋 ---
OBJECT_OFFSET_X = 0.0  # X 오프셋(좌/우)
OBJECT_OFFSET_Y = 0.0  # Y 오프셋(위/아래)
OBJECT_OFFSET_Z = 6.60 # Z 오프셋(앞/뒤)
OFFSET_STEP = 0.1      # 위치 조정 스텝

# --- 카메라 내재 파라미터 사용 여부 ---
USE_INTRINSICS = True  # True면 fx, fy, cx, cy로 P 행렬 구성(권장)

# --- 자동 스케일 맞춤(Auto-fit) ---
AUTO_FIT_ENABLE = True
AUTO_FIT_REF_PIX = 120.0   # 기준 얼굴 높이(px). 카메라 거리에 맞춰 스케일 자동 조정
AUTO_FIT_SMOOTH = 0.8      # 0.0(즉시) ~ 0.95(느리게)
HAT_SCALE_AT_REF = 1.0     # 기준 크기에서 갓 스케일
STRAP_SCALE_AT_REF = 1.0   # 기준 크기에서 끈 스케일

# -------------------- 표시/미러 설정 --------------------
SHOW_STRAP = True  # 끈 표시
SHOW_HAT = True    # 갓 표시
MIRROR_MODE = True
MIRROR_FIX_ON_MIRROR = True

# -------------------- 기본 오브젝트 파라미터 --------------------
o_params = dict(
    scale=1.0,
    tx=0.0,
    ty=0.0,
    tz=0.0,
    rx=0.0,
    ry=0.0,
    rz=0.0,
    mirror=False,
    string_scale=1.0,
)

# -------------------- 턱(하관) 오클루전 마스크 --------------------
JAW_OCCLUSION_ENABLED = True
JAW_DEBUG_SHOW = False
JAW_GROW_PX = 2
JAW_BLUR_SIGMA = 1.2
JAW_COLOR = (255, 64, 64)
JAW_ALPHA = 0.35
seg_thr = 0.85  # 세그 확률 임계

# -------------------- FaceMesh 얼굴 윤곽(oval) 인덱스 --------------------
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]

# -------------------- 물리(끈) 시뮬레이션 --------------------
MAX_BONES = 64
GRAVITY_ON = True
g_strength = -150.0   # 중력 가속도 (아래 방향/s^2)
ROPE_DAMP = 0.10      # 감쇠
ROPE_ITERS = 14       # 제약 반복 횟수

# 앵커 안정화 파라미터
ANCHOR_SMOOTHING = 0.15        # 앵커 위치 EMA 계수 (0=안함, 1=매우 느리게)
MAX_ANCHOR_DISTANCE = 50.0     # 프레임 간 앵커 최대 이동 허용(px)
STABILIZATION_FRAMES = 60      # 초기 안정화 프레임(예: 60FPS 기준 1초)
MAX_ROPE_VELOCITY = 100.0      # 로프 최대 속도 제한

# -------------------- 사람 세그멘트(마스크) --------------------
USE_SEG_MASK = True   # F8 토글
seg_thr_hat = 0.5     # 갓용 세그 임계

# -------------------- 갓 기본 변환(오브젝트 파라미터) --------------------
hat_base_params = dict(
    scale=1.0, tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0, mirror=False
)

# -------------------- 녹색 링(가이드) 파라미터 --------------------
green_radius = 6.85
green_half_h = 0.03
green_off_x = -0.20
green_off_y = -1.45
green_off_z = -1.20
green_segments = 64
green_scale = 1.15
green_pitch_deg = 0.0

# 녹색/파란 가이드 공통 위치(갓과 함께 사용)
green_tx = 0.0
green_ty = -0.50
green_tz = 0.0

# -------------------- 파란 화면(가이드) 파라미터 --------------------
blue_size = 20.0
blue_pitch_deg = 25.0
blue_alpha = 0.35
blue_color = (0.20, 0.60, 1.00)

blue_tx = 0.0
blue_ty = 0.0
blue_tz = 0.0

# -------------------- 갓 표시/오프셋/자세 파라미터 --------------------
hat_offset_x = 0.0
hat_offset_y = 0.0
hat_offset_z = 0.0
hat_scale = 1.0

OBJECT_PITCH_DEG = 0.0
OBJECT_YAW_DEG = 0.0
OBJECT_ROLL_DEG = 0.0

# -------------------- 디버그 색상 토글 --------------------
SHOW_GREEN_COLOR = False  # F11
SHOW_BLUE_COLOR = False   # F12
OCCLUSION_ENABLED = True  # F5

# -------------------- ONNX 세그먼트 모델 경로 --------------------
SELFIE_MODEL_PATH = "./asset/onnx/selfie_multiclass_256x256.onnx"
MODEL_INPUT_SIZE = (256, 256)

# -------------------- 2D 효과 주기/눈 깜빡임 --------------------
SEG_EVERY_N = 2  # 세그멘트 재계산 간격(N 프레임마다)

# 눈 뜸/감김 감지
EYE_OPEN_SMOOTH = 0.6
EYE_OPEN_THR_OPEN = 0.22
EYE_OPEN_THR_CLOSE = 0.18

# -------------------- 스킨 톤 보정 --------------------
SKIN_TONE_ENABLE = True
SKIN_TONE_TARGET = (139, 82, 110)  # BGR
SKIN_TONE_STRENGTH = 0.5
SKIN_TONE_BRIGHTNESS = 0.8
SKIN_TONE_FEATHER = 3.0

# --- ONNX 멀티클래스 설정 ---
# 스킨 클래스 ID 목록(모델 아웃풋 기준)
SKIN_CLASS_IDS = [2, 3]
# 스킨 확률 임계(0~1)
SKIN_PROB_THR = 0.4
# 스킨 확률 누적합 사용 여부(얇은 부위 보완)
USE_SKIN_PROB_SUM = True

# -------------------- 스모키 아이 --------------------
SMOKY_ENABLE = True
SMOKY_COLOR = (18, 18, 18)  # BGR
SMOKY_SIGMA = 7.0

# -------------------- 공막(흰자) 효과 --------------------
SCLERA_COLOR_ENABLE = True
SCLERA_COLOR = (30, 30, 30)  # BGR
SCLERA_ALPHA = 0.6
SCLERA_FEATHER = 1.5

# -------------------- 홍채/동공(눈) 효과 --------------------
EYE_ENABLE = True
IRIS_SCALE = 1.2
EYE_ALPHA = 0.9
EYE_EDGE_BLUR = 0.8
EYE_H_IN, EYE_S_IN, EYE_V_IN = 28, 220, 255   # 입력 HSV
EYE_H_OUT, EYE_S_OUT, EYE_V_OUT = 18, 180, 200  # 출력 HSV
PUPIL_SIZE = 0.60
PUPIL_THIN = 0.12
PUPIL_ALPHA = 0.98

# -------------------- 립 컬러 --------------------
LIP_COLOR_ENABLE = True
LIP_COLOR = (90, 2, 103)  # BGR
LIP_ALPHA = 0.65
LIP_FEATHER = 2.5
LIP_BRIGHTNESS = 0.7

# -------------------- 랜드마크 인덱스(눈/입/홍채) --------------------
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]
LIP_OUTER_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIP_INNER_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# 눈꺼풀/눈꼬리 보조 인덱스
LEFT_UP_IDX = [159, 158, 157]
LEFT_LO_IDX = [145, 144, 153]
LEFT_CORNERS = (33, 133)
RIGHT_UP_IDX = [386, 385, 384]
RIGHT_LO_IDX = [374, 373, 380]
RIGHT_CORNERS = (362, 263)

# -------------------- UV 스티커(얼굴 패턴) --------------------
UV_STICKER_ENABLE = True                 # 기능 활성화
UV_CSV = "./asset/face_uv_coords_fixed.csv"        # 고정된 UV 좌표 CSV
TRI_CSV = "./asset/mediapipe_triangles_478_calculated.csv"  # 삼각형 인덱스 CSV
PATTERN_RGBA = "./patterns/Untitled.png" # 적용할 패턴 RGBA 이미지
USE_STRAIGHT_ALPHA = True
ALPHA_GAIN = 1.0
WARP_FLAGS = cv2.INTER_LINEAR
UV_TRI_AREA_THR = 0.0  # 너무 작은 삼각형 스킵 임계(0.0=사용 안 함)

# -------------------- 배경 합성(옵션) --------------------
# 멀티클래스 ONNX/셀피 세그로 배경을 분리하고, 정지 이미지 또는 GIF로 대체 가능
USE_BG_IMAGE = False
BG_IMAGE_PATH = "./asset/WSG.gif"  # 배경 이미지 또는 .gif 경로
BG_CLASS_INDEX = 0                 # 멀티클래스에서 배경 클래스 인덱스(일반적으로 0)
BG_FIT_MODE = "cover"              # cover|contain|stretch
BG_GIF_PRELOAD = False
BG_SEG_EVERY_N = 2                 # 배경 인물 마스크 재계산 주기(N 프레임)
BG_FEATHER_SIGMA = 0.5             # 합성 경계 feather 강도(GaussianBlur sigma)
BG_MASK_SOFT = True                # 확률 기반 soft 마스크 사용
BG_MASK_CLOSE = 1                  # 형태학적 closing 반복(작은 구멍 메움)
BG_MASK_OPEN = 0                   # 형태학적 opening 반복(잡티 제거)
BG_MASK_GUIDED = False             # 가능 시 가이드 필터로 에지 정렬
BG_MASK_TEMPORAL_ALPHA = 0.15      # 마스크 템포럴 EMA 계수(0~1, 클수록 이전 프레임 가중↑)
BG_MASK_GAMMA = 1.0                # soft 경계 하드닝(gamma>1 → 배경 누수 감소)
BG_MASK_BIN_THR = 0                # 0=비활성, >0이면 임계치로 이진화(0~255)

# True: 모든 리소스 선로딩, False: 필요 시 로딩
# (주의: 메모리 상황에 따라 조정)

# ----- 얼굴 소유자 게이팅(가장 가까운 사람만 적용) -----
BG_GATE_TO_FACE = True
BG_GATE_BIN_THR = 96      # soft 마스크에서 CC 추출 임계
BG_GATE_DILATE = 1        # 선택된 성분 팽창 횟수
BG_GATE_MODE = "cc"       # 'cc'(연결요소) 또는 'bbox'(단순 바운딩박스)
# bbox 모드 시 여백(px)
BG_GATE_MARGIN_TOP_PX = 100
BG_GATE_MARGIN_BOTTOM_PX = 400   # 상체 포함을 위해 크게
BG_GATE_MARGIN_LEFT_PX = 150
BG_GATE_MARGIN_RIGHT_PX = 150
# 고급 게이트 파라미터
BG_GATE_THR_LOW = 48
BG_GATE_ROI_SCALE = 1.4
BG_GATE_CC_DILATE_RATIO = 0.5
BG_GATE_INCLUDE_NEAR = True
BG_GATE_NEAR_MULT = 3.0

# ----- Owner 선택(근접 기준) -----
OWNER_NUM_FACES = 1
OWNER_SWITCH_MARGIN = 1.15     # 경쟁자가 현재 면적의 배수 이상이면 전환
OWNER_SWITCH_FRAMES = 5        # 전환 유지 프레임
OWNER_TIMEOUT_FRAMES = 30      # 얼굴 미검출 시 소유자 리셋
OWNER_BOX_MARGIN_PX = 60

# 근접 판정(얼굴 높이 기준)
OWNER_NEAR_MIN_HEIGHT_PX = 120
OWNER_NEAR_MIN_RATIO = 0.18  # 0.18 * 720 = 129.6px → 131~138px 구간 통과

# --- 포즈(자세) 기반 게이팅(배경 세그용) ---
BG_POSE_GATE_ENABLE = True
POSE_GATE_MARGIN_TOP_PX = 200
POSE_GATE_MARGIN_BOTTOM_PX = 420
POSE_GATE_MARGIN_LEFT_PX = 140
POSE_GATE_MARGIN_RIGHT_PX = 140
POSE_OWNER_MAX_DIST_RATIO = 1.4  # 얼굴 폭 대비 거리 비율 한계

# -------------------- Background mask source selector --------------------
# 'onnx'  : 멀티클래스 ONNX 세그멘테이션 사용(기존 동작)
# 'selfie': MediaPipe SelfieSegmentation 실루엣 사용(빠르고 가벼움)
BG_MASK_SOURCE = "selfie"  # 'onnx' 또는 'selfie'
