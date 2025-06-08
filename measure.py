import cv2
import numpy as np
import mediapipe as mp
import math
from typing import Dict, Tuple

# －－－ 2. 核心函式 －－－
mp_pose        = mp.solutions.pose
mp_drawing     = mp.solutions.drawing_utils
POSE_LANDMARKS = mp_pose.PoseLandmark

def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """兩點像素距離"""
    return math.dist(p1, p2)

def _pixel2cm(px_len: float, px_ref: float, cm_ref: float) -> float:
    """透過『已知參考長度』把像素轉公分"""
    return px_len / px_ref * cm_ref

def measure_from_img(img_bgr: np.ndarray,
                     ref_method: str = "head",         # "head" 或 "custom"
                     cm_ref: float = 23.0,             # 若 ref_method="head"，預設頭長 23 cm
                     px_ref_custom: float = None       # 若 ref_method="custom"，請傳入像素長度
                    ) -> Dict:
    """
    主要量測函式：
    - 取得 Mediapipe Pose 33 點
    - 計算肩寬、骨盆寬、大腿長、小腿長
    - 回傳像素＋換算公分
    """
    with mp_pose.Pose(
            static_image_mode=True,
            enable_segmentation=False,
            model_complexity=1,
            min_detection_confidence=0.6) as pose:

        results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            raise ValueError("偵測失敗：沒找到人體骨骼點，請確認照片是否完整且光線充足。")

        h, w = img_bgr.shape[:2]
        # 把 landmark 座標轉成像素 (x,y)
        pts = { lm.name : (results.pose_landmarks.landmark[lm].x * w,
                           results.pose_landmarks.landmark[lm].y * h)
                for lm in POSE_LANDMARKS }

        # ========== 計算像素距離 ==========
        px_shoulder = _euclidean(pts['LEFT_SHOULDER'],  pts['RIGHT_SHOULDER'])
        px_hip      = _euclidean(pts['LEFT_HIP'],       pts['RIGHT_HIP'])
        px_thigh    = _euclidean(pts['LEFT_HIP'],       pts['LEFT_KNEE'])
        px_calf     = _euclidean(pts['LEFT_KNEE'],      pts['LEFT_ANKLE'])

        # 參考長度 (pixel)
        if ref_method == "head":
            px_ref = _euclidean(pts['LEFT_EAR'], pts['RIGHT_EAR'])   # 耳際寬當頭長 proxy
        elif ref_method == "custom":
            if px_ref_custom is None:
                raise ValueError("ref_method='custom' 時需提供 px_ref_custom")
            px_ref = px_ref_custom
        else:
            raise ValueError("ref_method 僅支援 'head' 或 'custom'")

        # ========== 換算公分 ==========
        cm_shoulder = _pixel2cm(px_shoulder, px_ref, cm_ref)
        cm_hip      = _pixel2cm(px_hip,      px_ref, cm_ref)
        cm_thigh    = _pixel2cm(px_thigh,    px_ref, cm_ref)
        cm_calf     = _pixel2cm(px_calf,     px_ref, cm_ref)

        # ========== 組結果 ==========
        return {
            "pixel": {
                "shoulder_width": px_shoulder,
                "hip_width"     : px_hip,
                "thigh_length"  : px_thigh,
                "calf_length"   : px_calf,
                "ref_length_px" : px_ref
            },
            "cm": {
                "shoulder_width": round(cm_shoulder, 1),
                "hip_width"     : round(cm_hip, 1),
                "thigh_length"  : round(cm_thigh, 1),
                "calf_length"   : round(cm_calf, 1),
                "ref_length_cm" : cm_ref
            }
        }

def draw_landmarks(img_bgr: np.ndarray, results) -> np.ndarray:
    """輔助：在影像上畫骨架（可視化用）"""
    annotated = img_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2))
    return annotated

# －－－ 3. 簡易 Demo (上傳 → 量測) －－－
from google.colab import files
uploaded = files.upload()            # 會跳出選檔視窗

for fname in uploaded.keys():
    img = cv2.imread(fname)
    try:
        info = measure_from_img(img, ref_method="head")  # 或改成 custom
    except ValueError as e:
        print(e)
        continue

    # 顯示結果
    print(f"\n📝 {fname} 量測結果（公分）:")
    for k, v in info["cm"].items():
        if k != "ref_length_cm":
            print(f" - {k:>14s}: {v:>5.1f} cm")

    # 視覺化
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    annotated = draw_landmarks(img, res)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.show()
