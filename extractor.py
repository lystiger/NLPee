# -*- coding: utf-8 -*-
"""
extractor.py
------------
Trích xuất đặc trưng 2 tay từ video dùng MediaPipe trực tiếp (không cần YOLO).
Mỗi frame → 126 features: tay trái (63) + tay phải (63).
Nếu thiếu 1 tay → pad zeros cho tay đó.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FEATURE_DIM = 126   # 21 keypoints × 3 (x,y,z) × 2 tay
HAND_DIM    = 63    # features cho 1 tay
SEQ_LENGTH  = 35    # số frames chuẩn (giữ nguyên như code cũ)


# ─────────────────────────────────────────────
# KEYFRAME EXTRACTOR
# ─────────────────────────────────────────────
class AdaptiveKeyframeExtractor:
    """Lấy đúng SEQ_LENGTH frames từ video, lặp frame nếu video ngắn hơn."""

    def __init__(self, target_frames=SEQ_LENGTH):
        self.target_frames = target_frames

    def extract_frame_indices(self, video_path):
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total <= 0:
            return []

        if total >= self.target_frames:
            indices = np.linspace(0, total - 1, self.target_frames, dtype=int).tolist()
        else:
            indices = list(range(total))
            while len(indices) < self.target_frames:
                indices.append(indices[-1])

        return indices[:self.target_frames]


# ─────────────────────────────────────────────
# HAND KEYPOINT EXTRACTOR (chỉ dùng MediaPipe)
# ─────────────────────────────────────────────
class HandKeypointExtractor:
    """
    Dùng MediaPipe Hands trực tiếp trên toàn frame.
    Không cần YOLO, không cần file hand.pt.

    Output mỗi frame: vector 126 chiều
      [0:63]   = tay trái  (zeros nếu không thấy)
      [63:126] = tay phải  (zeros nếu không thấy)
    """

    def __init__(self):
        print("Initializing MediaPipe Hands...")
        try:
            import mediapipe as mp  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Thiếu dependency `mediapipe`. Cài bằng `python -m pip install -r requirements.txt`."
            ) from e

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Gói `mediapipe` hiện tại không có `solutions` (thường do cài nhầm package hoặc bị "
                "trùng tên file/thư mục `mediapipe`). Thử: `python -m pip uninstall mediapipe -y` rồi "
                "`python -m pip install mediapipe`."
            )

        try:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            raise RuntimeError(f"Không khởi tạo được MediaPipe Hands: {e}") from e
        self.feature_dim = FEATURE_DIM

    def _landmarks_to_array(self, hand_landmarks):
        kps = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        return np.array(kps, dtype=np.float32).flatten()

    def extract(self, frame):
        left_feat  = np.zeros(HAND_DIM, dtype=np.float32)
        right_feat = np.zeros(HAND_DIM, dtype=np.float32)

        try:
            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = self.hands.process(frame_rgb)

            if mp_results.multi_hand_landmarks is None:
                return np.concatenate([left_feat, right_feat])

            for landmarks, handedness in zip(
                mp_results.multi_hand_landmarks,
                mp_results.multi_handedness,
            ):
                label = handedness.classification[0].label  # 'Left' hoặc 'Right'
                feat  = self._landmarks_to_array(landmarks)
                if label == 'Left':
                    left_feat  = feat
                else:
                    right_feat = feat

        except Exception as e:
            print(f"[ExtractError] {e}")

        return np.concatenate([left_feat, right_feat])

    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()


# ─────────────────────────────────────────────
# FEATURE CACHE
# ─────────────────────────────────────────────
class VideoFeatureCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, video_path):
        return self.cache_dir / (Path(video_path).stem + '.npy')

    def exists(self, video_path):
        return self._path(video_path).exists()

    def save(self, video_path, arr):
        np.save(str(self._path(video_path)), arr)

    def load(self, video_path):
        p = self._path(video_path)
        if not p.exists():
            return None
        return np.load(str(p))


# ─────────────────────────────────────────────
# PRECOMPUTE (BATCH)
# ─────────────────────────────────────────────
def precompute_features(videos, extractor, cache, seq_length=SEQ_LENGTH):
    to_process = [v for v in videos if not cache.exists(v)]
    if not to_process:
        print("Tất cả features đã được cache!")
        return

    print(f"Đang xử lý {len(to_process)} videos...")
    kf_extractor = AdaptiveKeyframeExtractor(target_frames=seq_length)

    for video_path in tqdm(to_process, desc="Extracting features"):
        try:
            indices = kf_extractor.extract_frame_indices(video_path)
            cap     = cv2.VideoCapture(video_path)
            feats   = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    feats.append(np.zeros(FEATURE_DIM, dtype=np.float32))
                    continue
                feats.append(extractor.extract(frame))

            cap.release()

            while len(feats) < seq_length:
                feats.append(
                    feats[-1].copy() if feats
                    else np.zeros(FEATURE_DIM, dtype=np.float32)
                )

            arr = np.stack(feats[:seq_length], axis=0).astype(np.float32)
            cache.save(video_path, arr)

        except Exception as e:
            print(f"\n[Lỗi] {video_path}: {e}")
            cache.save(video_path, np.zeros((seq_length, FEATURE_DIM), dtype=np.float32))

    print("Trích xuất features hoàn tất.")
