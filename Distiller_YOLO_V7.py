"""
Distiller_YOLO_V7.py
====================
V7 Data Collection Script — Teacher-Student Distillation

Downloads 11 traffic videos from YouTube at 720p quality using yt-dlp,
then uses the full YOLOv8n model (the "teacher") to auto-label every Nth frame
with YOLO-format bounding boxes. These labeled frames form the training dataset
for the V7 student model.

Output structure:
    dataset_yolo/
        images/train/   ← 80% of frames (JPG)
        images/val/     ← 20% of frames (JPG)
        labels/train/   ← matching YOLO .txt label files
        labels/val/
"""

import cv2
import os
import numpy as np
import subprocess
import random
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────

# The 11 YouTube video IDs from the traffic-camera playlist.
# These were chosen for maximum scene diversity:
#   Highway (straight), Intersection (high-angle), Roundabout, Dusk/Low-light,
#   Dense urban slow traffic.
VIDEO_IDS = [
    "wqctLW0Hb_0",
    "QuUxHIVUoaY",
    "TW3EH4cnFZo",
    "KBsqQez-O4w",
    "nt3D26lrkho",
    "PNCJQkvALVc",
    "BLc8s-_tsiQ",
    "eCQoTgxCCSg",
    "MNn9qKG2UFI",
    "wWLAc6mdJrs",
    "MZNbLWtJgnw",
]

VIDEO_DIR       = 'videos_hq'       # Where downloaded 720p mp4s are stored
OUTPUT_DIR      = 'dataset_yolo'    # Final YOLO-format dataset
FRAME_STEP      = 30                # Sample 1 frame every 30 frames (~1 fps at 30fps source)
MAX_PER_VIDEO   = 300               # Cap frames per video to balance the dataset
CONF_THRESH     = 0.40              # Minimum teacher confidence to accept a label
VAL_SPLIT       = 0.20              # 20% of frames go to validation set

# COCO class IDs we treat as "vehicle"
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = {2, 3, 5, 7}


def download_video(video_id: str, out_dir: str) -> str | None:
    """
    Download a single YouTube video at 720p using yt-dlp.
    Returns the path to the downloaded file, or None on failure.
    Uses yt-dlp format selector to prefer 720p mp4 with audio stripped.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_id}.mp4")

    if os.path.exists(out_path):
        print(f"  Already downloaded: {video_id}")
        return out_path

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", out_path,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    print(f"  Downloading {video_id} at 720p...")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  WARNING: yt-dlp failed for {video_id}")
        return None
    return out_path


def extract_labeled_frames(video_path: str, teacher: YOLO) -> list[tuple]:
    """
    Process a video file frame-by-frame.
    For each sampled frame, run the teacher YOLO model and collect boxes
    for COCO vehicle classes.

    Returns a list of (frame_bgr, labels_list) tuples where labels_list
    contains YOLO-format strings: "0 cx cy w h"
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    samples = []
    frame_idx = 0

    while len(samples) < MAX_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            h, w = frame.shape[:2]
            results = teacher(frame, verbose=False)[0]
            boxes_data = results.boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

            labels = []
            for box in boxes_data:
                if len(box) < 6:
                    continue
                x1, y1, x2, y2, conf, cls = box
                if conf >= CONF_THRESH and int(cls) in VEHICLE_CLASSES:
                    # Convert absolute pixel coords → YOLO normalised format
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    # Class 0 = "vehicle" (we merge all vehicle types into one class)
                    labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if labels:
                samples.append((frame.copy(), labels))

        frame_idx += 1

    cap.release()
    return samples


def save_dataset(samples: list[tuple], split: str) -> None:
    """Save (frame, labels) pairs into the YOLO dataset directory structure."""
    img_dir = os.path.join(OUTPUT_DIR, 'images', split)
    lbl_dir = os.path.join(OUTPUT_DIR, 'labels', split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i, (frame, labels) in enumerate(samples):
        ts = f"{split}_{i:06d}"
        cv2.imwrite(os.path.join(img_dir, f"{ts}.jpg"), frame)
        with open(os.path.join(lbl_dir, f"{ts}.txt"), 'w') as f:
            f.write("\n".join(labels))


def main():
    print("=== Distiller_YOLO_V7.py — V7 Dataset Collection ===")
    print(f"Videos : {len(VIDEO_IDS)}")
    print(f"Output : {OUTPUT_DIR}/")
    print()

    # Load teacher model (full YOLOv8n, COCO pretrained)
    print("Loading teacher model (YOLOv8n COCO)...")
    teacher = YOLO('yolov8n.pt')

    all_train, all_val = [], []

    for vid_id in VIDEO_IDS:
        print(f"\n[{vid_id}]")
        video_path = download_video(vid_id, VIDEO_DIR)
        if video_path is None:
            continue

        samples = extract_labeled_frames(video_path, teacher)
        print(f"  Labeled frames collected: {len(samples)}")

        # Split into train/val
        random.shuffle(samples)
        n_val = max(1, int(len(samples) * VAL_SPLIT))
        all_val.extend(samples[:n_val])
        all_train.extend(samples[n_val:])

    print(f"\nTotal — Train: {len(all_train)}  Val: {len(all_val)}")

    print("Saving dataset...")
    save_dataset(all_train, 'train')
    save_dataset(all_val, 'val')

    # Write data YAML
    yaml_path = 'v7_data.yaml'
    dataset_abs = os.path.abspath(OUTPUT_DIR).replace('\\', '/')
    with open(yaml_path, 'w') as f:
        f.write(f"train: {dataset_abs}/images/train\n")
        f.write(f"val:   {dataset_abs}/images/val\n\n")
        f.write("nc: 1\n")
        f.write("names: ['vehicle']\n")

    print(f"\nDone. Dataset YAML written to {yaml_path}")
    print("Next step: python Train_YOLO_V7.py")


if __name__ == "__main__":
    main()
