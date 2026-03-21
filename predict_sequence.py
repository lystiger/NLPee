# -*- coding: utf-8 -*-
"""
predict_sequence.py
-------------------
Dự đoán chuỗi 2–3 từ từ 1 video chứa nhiều cử chỉ liên tiếp.
Thay thế sequence.py / sequence2.py / sequence3.py từ code cũ.
"""

import os
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

from extractor import HandKeypointExtractor, FEATURE_DIM, SEQ_LENGTH
from model_bundle import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, load_model
from predict_word import CLASSES


def extract_segments_from_video(video_path, extractor, silence_threshold=10):
    """
    Đọc video, trích xuất features từng frame, tách thành các đoạn
    dựa trên vùng im lặng (toàn zeros) giữa các cử chỉ.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    all_feats = []
    for _ in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        all_feats.append(extractor.extract(frame))
    cap.release()

    if not all_feats:
        return []

    all_feats = np.stack(all_feats, axis=0)          # (T, 126)
    is_silent = np.all(all_feats == 0, axis=1)        # (T,)

    segments = []
    i = 0
    T = len(is_silent)

    while i < T:
        while i < T and is_silent[i]:
            i += 1
        if i >= T:
            break

        j = i
        silence_count = 0
        while j < T:
            if is_silent[j]:
                silence_count += 1
                if silence_count >= silence_threshold:
                    break
            else:
                silence_count = 0
            j += 1

        seg_end = j - silence_count
        segment = all_feats[i:seg_end]
        if len(segment) > 0:
            segments.append(segment)
        i = j

    return segments


def pad_or_trim(segment, seq_length=SEQ_LENGTH):
    T = segment.shape[0]
    if T >= seq_length:
        indices = np.linspace(0, T - 1, seq_length, dtype=int)
        return segment[indices]
    else:
        pad = np.repeat(segment[-1:], seq_length - T, axis=0)
        return np.concatenate([segment, pad], axis=0)


def predict_sequence(video_path, model_path, output_path=None,
                     true_label=None, silence_threshold=10):
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _  = load_model(model_path=model_path, metadata_path=DEFAULT_METADATA_PATH, device=device)
    extractor = HandKeypointExtractor()

    print(f"Đang xử lý: {video_path}")
    segments = extract_segments_from_video(video_path, extractor, silence_threshold)
    del extractor

    if not segments:
        print("[Cảnh báo] Không phát hiện cử chỉ nào trong video.")
        return ""

    print(f"Phát hiện {len(segments)} cử chỉ.")

    predicted_words = []
    with torch.no_grad():
        for i, seg in enumerate(segments):
            feat     = pad_or_trim(seg, SEQ_LENGTH).astype(np.float32)
            x        = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            out      = model(x)
            pred_idx = out.argmax(1).item()
            predicted_words.append(CLASSES[pred_idx])
            print(f"  Đoạn {i+1}: {CLASSES[pred_idx]}")

    result = ' '.join(predicted_words)
    print(f"\nKết quả: {result}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            if true_label:
                f.write(f"true_label: {true_label} === ")
            f.write(f"Predict sequence: {result}\n")
        print(f"Lưu kết quả tại: {output_path}")

    return result


def predict_sequence_batch(data_dir, model_path, output_path, silence_threshold=10):
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _  = load_model(model_path=model_path, metadata_path=DEFAULT_METADATA_PATH, device=device)
    extractor = HandKeypointExtractor()

    results = []
    for label_dir in sorted(Path(data_dir).iterdir()):
        if not label_dir.is_dir():
            continue
        true_label = label_dir.name
        for vf in sorted(label_dir.glob('*.mp4')):
            segments = extract_segments_from_video(str(vf), extractor, silence_threshold)
            if not segments:
                results.append((true_label, ''))
                continue

            predicted_words = []
            with torch.no_grad():
                for seg in segments:
                    feat     = pad_or_trim(seg, SEQ_LENGTH).astype(np.float32)
                    x        = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
                    out      = model(x)
                    predicted_words.append(CLASSES[out.argmax(1).item()])

            results.append((true_label, ' '.join(predicted_words)))

    del extractor

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for true_label, pred in results:
            f.write(f"true_label: {true_label} === Predict sequence: {pred}\n")

    print(f"Kết quả lưu tại: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Sign Language Sequence')
    parser.add_argument('--video_path',        help='Đường dẫn 1 video')
    parser.add_argument('--data_dir',          help='Thư mục batch mode')
    parser.add_argument('--model_path',        default=str(DEFAULT_MODEL_PATH))
    parser.add_argument('--output_path',       default='output/sequence.txt')
    parser.add_argument('--true_label',        default=None)
    parser.add_argument('--silence_threshold', type=int, default=10)
    args = parser.parse_args()

    if args.video_path:
        predict_sequence(
            video_path        = args.video_path,
            model_path        = args.model_path,
            output_path       = args.output_path,
            true_label        = args.true_label,
            silence_threshold = args.silence_threshold,
        )
    elif args.data_dir:
        predict_sequence_batch(
            data_dir          = args.data_dir,
            model_path        = args.model_path,
            output_path       = args.output_path,
            silence_threshold = args.silence_threshold,
        )
    else:
        print("Cần cung cấp --video_path hoặc --data_dir")
