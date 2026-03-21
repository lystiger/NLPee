# -*- coding: utf-8 -*-
"""
predict_word.py
---------------
Dự đoán 1 từ từ video. Thay thế predictname.py từ code cũ.
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path

from extractor import HandKeypointExtractor, VideoFeatureCache, precompute_features, FEATURE_DIM, SEQ_LENGTH
from model_bundle import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH, get_labels, load_model


CLASSES = get_labels()


def predict_from_videos(data_dir, model_path, output_path, cache_dir='cache/predict/'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(model_path=model_path, metadata_path=DEFAULT_METADATA_PATH, device=device)

    vids, true_labels = [], []
    for ci, cname in enumerate(CLASSES):
        p = Path(data_dir) / cname
        if not p.exists():
            continue
        for f in list(p.glob('*.mp4')) + list(p.glob('*.avi')) + list(p.glob('*.MOV')):
            vids.append(str(f))
            true_labels.append(cname)

    if not vids:
        print(f"[Lỗi] Không tìm thấy video trong {data_dir}")
        return

    extractor = HandKeypointExtractor()
    cache     = VideoFeatureCache(cache_dir)
    precompute_features(vids, extractor, cache, seq_length=SEQ_LENGTH)
    del extractor

    correct = 0
    results = []

    with torch.no_grad():
        for vid, true_label in zip(vids, true_labels):
            feat = cache.load(vid)
            if feat is None:
                feat = np.zeros((SEQ_LENGTH, FEATURE_DIM), dtype=np.float32)

            x        = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            out      = model(x)
            pred_idx = out.argmax(1).item()
            pred_label = CLASSES[pred_idx]

            if pred_label == true_label:
                correct += 1
            results.append((true_label, pred_label))
            print(f"True: {true_label:20s} → Pred: {pred_label}")

    acc = correct / len(results) * 100
    print(f"\nAccuracy: {correct}/{len(results)} = {acc:.2f}%")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for true_label, pred_label in results:
            f.write(f"true_label: {true_label} === predict_label: {pred_label}\n")
        f.write(f"\nAccuracy: {acc:.2f}%\n")

    print(f"Kết quả lưu tại: {output_path}")


def predict_single_video(video_path, model_path, cache_dir='cache/predict/'):
    """Trả về nhãn dự đoán cho 1 video duy nhất. Dùng trong GUI."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(model_path=model_path, metadata_path=DEFAULT_METADATA_PATH, device=device)

    extractor = HandKeypointExtractor()
    cache     = VideoFeatureCache(cache_dir)
    precompute_features([video_path], extractor, cache, seq_length=SEQ_LENGTH)
    del extractor

    feat = cache.load(video_path)
    if feat is None:
        return "Không phát hiện được tay"

    with torch.no_grad():
        x        = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
        out      = model(x)
        pred_idx = out.argmax(1).item()

    return CLASSES[pred_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Sign Language (1 từ)')
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--model_path',  default=str(DEFAULT_MODEL_PATH))
    parser.add_argument('--output_path', default='output/result.txt')
    parser.add_argument('--cache_dir',   default='cache/predict/')
    args = parser.parse_args()

    predict_from_videos(
        data_dir    = args.data_dir,
        model_path  = args.model_path,
        output_path = args.output_path,
        cache_dir   = args.cache_dir,
    )
