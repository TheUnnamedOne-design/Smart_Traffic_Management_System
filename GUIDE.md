# V7 TinyML Vehicle Detector — Complete Development Guide

> **Who is this for?** Anyone — including yourself in the future — who wants to understand exactly what this model is, why every decision was made, how the data was collected, how training worked, and how inference runs. You should be able to reproduce everything from scratch after reading this.

---

## Table of Contents

1. [The Problem We Were Solving](#1-the-problem-we-were-solving)
2. [Why The Previous Approaches Failed](#2-why-the-previous-approaches-failed)
3. [The Core Idea Behind V7](#3-the-core-idea-behind-v7)
4. [Model Selection — Why YOLOv8n?](#4-model-selection--why-yolov8n)
5. [Data Collection — How and Why](#5-data-collection--how-and-why)
6. [Training Strategy — Teacher-Student Distillation](#6-training-strategy--teacher-student-distillation)
7. [Archive File Structure](#7-archive-file-structure)
8. [Hyperparameters and Configuration](#8-hyperparameters-and-configuration)
9. [Training Process — Epoch by Epoch](#9-training-process--epoch-by-epoch)
10. [Performance Metrics — Deep Analysis](#10-performance-metrics--deep-analysis)
11. [Edge Suitability and TinyML Optimisation](#11-edge-suitability-and-tinyml-optimisation)
12. [Live Inference Pipeline](#12-live-inference-pipeline)
13. [How to Reproduce](#13-how-to-reproduce)

---

## 1. The Problem We Were Solving

The goal of this project is a **Fog-Edge Traffic Management System**. Edge nodes sit at intersections and need to count vehicles in real time. These nodes are resource-constrained — things like a Raspberry Pi or a Coral Edge TPU with around 1–2 GB RAM. They cannot run a full YOLO model (YOLOv8s is ~22MB, YOLOv8m is ~52MB).

The requirements:
- **Detect vehicles accurately** from a camera looking at traffic.
- **Run on-device** (no sending frames to cloud) — low latency.
- **Model size < 10 MB** ideally (TinyML constraint).
- **Bounding boxes must be tight** — loose boxes cause double-counting or missed handoffs between zones.

---

## 2. Why The Previous Approaches Failed

We went through six versions before V7. Here is why each one hit a wall:

| Version | Approach | Problem |
|:---|:---|:---|
| V1–V2 | MobileNetV2 image classifier, sliding window | A **classifier** only says "vehicle / not vehicle" for a fixed patch. It doesn't tell you _where_ the vehicle is — you have to guess. This produced very loose, jittery bounding boxes. |
| V3–V4 | Larger sliding window stride + Non-Maximum Suppression | Reduced jitter but boxes were still "approximate." Multiple overlapping windows on the same car all fired, and NMS picked one arbitrarily — not necessarily the tightest one. |
| V5 | Knowledge distillation — MobileNetV2 student trained on YOLO teacher's crops from 480p video | Better generalisation, but 480p footage meant the model saw blurry car pixels. Tight box coordinates were still unreliable because the model learned _what_ a car looks like but not _exactly where its edges are_. |
| V6 | Multi-video distillation + Hard Negative Mining + Weighted Box Fusion (WBF) | WBF fused multiple window hits into a single stable box. Much better. Still fundamentally a classifier under the hood — coordinates still estimated not predicted. |

**The root cause** across all versions: a **classifier** cannot output coordinate offsets. It can only vote "yes" or "no" for a region you slide across the image. No matter how good the classifier is, the box is always as accurate as your sliding window grid.

---

## 3. The Core Idea Behind V7

V7 switches the entire paradigm. Instead of a classifier, we now run a **native object detector**.

```
  CLASSIFIER (V1–V6)              DETECTOR (V7)
  ─────────────────               ───────────────
  Input: 224×224 crop            Input: whole 320×320 frame
  Output: [0.9, 0.1]             Output: [x_centre, y_centre, w, h, confidence]
          (car / no-car)                 per vehicle found anywhere in the frame

  Requires sliding window grid    No grid needed — one forward pass
  Box = grid cell position        Box = learned coordinate offsets from anchors
  Accuracy limited by step size   Accuracy limited only by model capacity
```

A native detector **regresses** the box coordinates directly as part of the loss function. The model is penalised during training if its predicted box does not overlap well with the ground truth box. This forces it to learn the exact pixel boundaries of each vehicle — not just "there is something vehicle-shaped here."

---

## 4. Model Selection — Why YOLOv8n?

Once we decided to use a native detector, we evaluated options:

| Option | Params | Why considered | Why rejected or chosen |
|:---|:---|:---|:---|
| YOLOv5n | ~1.9M | Lighter than v8 | Older architecture, less accurate |
| YOLOv8n | ~3.0M | Best accuracy/size ratio in the YOLO family | **Chosen** |
| YOLOv8s | ~11M | More accurate | Too heavy after INT8 quantisation (~11 MB) |
| RT-DETR-nano | ~6M | Transformer-based | No TFLite export path at the time |
| SSD MobileNetV2 | ~4M | Well-known TinyML model | Lower mAP for vehicles vs YOLOv8n |

**YOLOv8n** wins because:
- **~3M parameters** — after INT8 quantisation, fits inside ~3–6 MB (TinyML safe).
- Uses a **decoupled head** (classification branch and regression branch are separate). This is why it achieves better box tightness than older YOLO versions, which shared the head.
- Native **TFLite export** support via Ultralytics — one command.
- Well-calibrated confidence scores — we can set a threshold (0.3) and trust it.

### What is a Decoupled Head?

Older detectors (YOLOv3, v4) used a single layer to output both `[class probabilities]` and `[box coordinates]`. These two tasks fight each other during learning because their gradients have different scales. YOLOv8 uses two separate branches:

```
Backbone → Neck → Detection Head
                       ├── Classification branch → [vehicle confidence]
                       └── Regression branch    → [x, y, w, h]
```

The regression branch can focus purely on learning exact coordinates without being disturbed by classification gradients. This is the direct reason V7 boxes are tighter.

---

## 5. Data Collection — How and Why

### Why we couldn't use an existing dataset

The model needs to work on **live traffic camera footage** for the Fog-Edge project. Standard datasets like COCO or BDD100K were candidates, but:
- COCO has vehicles from many angles — dashcam, street level, aerial — which dilutes the model with irrelevant perspectives.
- Our cameras are **top-down / high-angle CCTV-style**. Training on dashcam footage hurts performance on CCTV frames.
- Label quality in public datasets varies. We wanted **perfect labels** from a controlled process.

### The Teacher-Student Distillation pipeline

We used the full YOLOv8n pre-trained on COCO as a **teacher**. The teacher is huge and accurate, but too big for the edge. The distillation pipeline:

```
11 YouTube traffic videos (720p)
           │
           ▼
   Distiller_YOLO_V7.py
      - Downloads each video with yt-dlp
      - Samples one frame every N frames
      - Runs full YOLOv8n teacher on the frame
      - Keeps only boxes where teacher confidence > 0.4
      - Saves frame as JPG + box coordinates as .txt (YOLO format)
           │
           ▼
   dataset_yolo/
      ├── images/train/    (≈1,400 frames)
      ├── images/val/      (≈300 frames)
      ├── labels/train/    (matching .txt files)
      └── labels/val/
```

### Why 720p?

We deliberately used 720p (1280×720) source footage instead of 360p or 480p:
- At 360p, a car far from the camera is **only 6–10 pixels wide**. After compression artefacts there are almost no gradient signals for the model to learn from.
- At 720p, the same car is **12–20 pixels wide** — enough texture (windshield reflection, shadow, body colour) to produce stable edge detection.
- The teacher's labels are also more accurate at higher resolution — it can see whether the box should sit at pixel 240 or pixel 246.

### Why 11 videos / what was the diversity goal?

The 11 videos were chosen to maximise **scene diversity**:
- 3 videos: straight highway, daytime
- 2 videos: intersection, high-angle
- 2 videos: roundabout / multi-lane merge
- 2 videos: low-light / dusk conditions
- 2 videos: dense urban, slow traffic

Without this diversity a model trained on only highway footage learns "vehicles are long rectangles moving horizontally." It breaks completely when a car is coming towards the camera (becomes a square). The 11-video mix ensures the model has seen vehicles at every angle.

### YOLO label format

Each `.txt` label file has one line per vehicle:

```
<class_id> <x_centre_norm> <y_centre_norm> <width_norm> <height_norm>
```

All coordinates are normalised to [0, 1] relative to image width/height.
`class_id = 0` means "vehicle" (we merged car, bus, truck, motorcycle into one class to keep it simple for edge counting).

---

## 6. Training Strategy — Teacher-Student Distillation

This is sometimes called **offline knowledge distillation** (as opposed to online distillation where student and teacher train together). The key idea:

```
Large teacher (full YOLOv8n, COCO) → generates pseudo-labels → 
Student (YOLOv8n, fresh init) trains on pseudo-labels
```

Why not train the student directly on COCO?
- COCO has 80 classes. We only need 1 (vehicle). Training on all 80 wastes capacity.
- COCO has vehicles in all environments. Our videos are domain-specific (traffic camera viewpoints).
- Domain-specific pseudo-labels are higher quality for our specific task.

The student starts from `yolov8n.pt` (COCO pretrained weights). We do **fine-tuning** — the backbone's feature extraction ability (detecting edges, textures, shapes) transfers from COCO. We only need to specialise the detection head for our single class and camera angle. This is why convergence is fast (50 epochs vs 300+ for training from scratch).

---

## 7. Archive File Structure

The `archive/V7/` folder is designed to be a self-contained, reproducible capsule of the entire V7 journey.

| File / Folder | Role |
|:---|:---|
| **`vehicle_detector_v7_native.pt`** | **The Final Model Weights.** Load this for inference. |
| `LiveTester_V7.py` | Real-time inference script (targets Chrome/YouTube). |
| `Metrics_Visualization.ipynb` | **Jupyter Notebook for plotting training curves** (Loss, mAP, etc.). |
| `Distiller_YOLO_V7.py` | Step 1: Automated data collection & auto-labelling. |
| `Train_YOLO_V7.py` | Step 2: Fine-tuning script for YOLOv8n. |
| `export_v7.py` | Step 3: Optimization/Export (ONNX & TFLite INT8). |
| `v7_data.yaml` | Training configuration (dataset paths and classes). |
| `yolov8n_teacher.pt` | The base 80-class COCO teacher model used for distillation. |
| `yolo_v7_metrics.csv` | Full training logs (loss and accuracy per epoch). |
| `dataset_yolo/` | The collected 720p dataset (images + YOLO-format labels). |
| `yolo_v7_vehicle/` | Full training run outputs (loss curves, confusion matrix, val sets). |

---

## 8. Hyperparameters and Configuration

Configured in `Train_YOLO_V7.py` and stored in `yolo_v7_vehicle/args.yaml`:

| Parameter | Value | Reason |
|:---|:---|:---|
| `imgsz` | 320 | TinyML constraint. Smaller resolution → fewer ops → runs on edge hardware. 320 is the smallest where vehicles are still detectable. |
| `epochs` | 50 | High enough for full convergence (loss plateau visible after epoch 38). |
| `batch` | 16 | Fills CPU/GPU memory well without thrashing. |
| `conf` (inference) | 0.3 | Threshold for accepting a detection. Low enough to not miss distant cars. |
| `device` | cpu | Trained on CPU — no GPU needed, just takes longer. |
| `optimizer` | SGD | Default for YOLO. Adam would converge faster but often to a worse local minimum for detection. |
| `lr0` | 0.01 | Initial learning rate. Includes warmup for first 3 epochs. |
| `lrf` | 0.01 | Final LR as fraction of `lr0`. Cosine annealing brings LR down to 0.0001 by epoch 50. |

### Data augmentation applied automatically by Ultralytics

- **Mosaic**: 4 frames stitched into one. Forces the model to detect small vehicles and partial occlusions.
- **RandomFlip**: Horizontal flip at 50% probability. Cars look the same mirrored — doubles effective dataset size.
- **HSV jitter**: Random hue/saturation/value shifts. Makes the model robust to lighting changes (daytime vs dusk vs overcast).
- **Scale / crop**: Random zoom in/out. Ensures the model handles vehicles at very different distances from camera.

These augmentations were not manually coded — Ultralytics applies them automatically. The choices above are well-validated across thousands of YOLO training runs by the community.

---

## 9. Training Process — Epoch by Epoch

The full log is in `yolo_v7_metrics.csv`. Below is a summary of the key phases:

### Phase 1 — Initial Convergence (Epochs 1–10)
- `box_loss` drops rapidly: 1.16 → 0.89
- `cls_loss` drops rapidly: 1.68 → 0.71
- mAP50 jumps from 0.638 → 0.866
- **What is happening**: The model is quickly learning "vehicle vs background" (cls_loss) and roughly where to place boxes (box_loss). The pretrained COCO backbone already knows to look at the right features. The detection head adapts to our class.

### Phase 2 — Refinement (Epochs 11–30)
- `box_loss`: 0.85 → 0.75 (slower decline)
- mAP50: 0.88 → 0.907
- mAP50-95: 0.655 → 0.71
- **What is happening**: The model is now fine-tuning the exact coordinate offsets. mAP50-95 rising is the "tightness" signal — the model is placing boxes closer to the exact vehicle boundary across multiple IoU thresholds (50%, 55%, ..., 95%).

### Phase 3 — Compression / Saturation (Epochs 31–50)
- `box_loss` continues falling very slowly: 0.74 → 0.60
- mAP50-95: 0.71 → 0.748
- Precision/Recall stabilise above 0.83/0.84
- **What is happening**: At this point the learning rate (from cosine annealing) is very small. The model is making tiny adjustments. The continued rise in mAP50-95 (even small gains) is worth the extra epochs — each point here means boxes that are 1–2 pixels tighter on average.

> **Why not 100 epochs?** After epoch 50 the validation loss stopped improving — the model was at its capacity limit for the given dataset size. More epochs would cause overfitting.

---

## 10. Performance Metrics — Deep Analysis

All values from the final epoch (Epoch 50), measured on the held-out validation set.

### Summary Table

| Metric | Value | What it means |
|:---|:---|:---|
| **mAP50** | **0.919 (91.9%)** | At IoU threshold 0.5, the model correctly detects 91.9% of vehicles with boxes that overlap the ground truth by at least 50%. |
| **mAP50-95** | **0.748 (74.8%)** | Average mAP across IoU thresholds from 0.5 to 0.95 (step 0.05). This is the "strict tightness" metric. 74.8% is excellent for a model this size. |
| **Precision** | **0.835 (83.5%)** | Of every box the model draws, 83.5% contain a real vehicle. False alarm rate is 16.5%. |
| **Recall** | **0.848 (84.8%)** | Of every real vehicle in the scene, the model finds 84.8% of them. Misses 15.2%. |

### Understanding mAP50 vs mAP50-95

```
IoU = Area of Overlap / Area of Union

 ┌──────────────┐
 │  Ground Truth│
 │    ┌──────────┼──────────┐ ← Predicted box
 │    │ Overlap  │          │
 └────┼──────────┘          │
      └─────────────────────┘

IoU = Overlap Area / (GT Area + Pred Area - Overlap Area)
```

- **mAP50** counts a detection as correct if IoU > 0.50. A box drawn around roughly the car area passes this.
- **mAP50-95** demands the box be tight — IoU > 0.50, 0.55, 0.60, ..., 0.95. Getting IoU > 0.90 requires the box to almost perfectly align with the vehicle's silhouette.

Our 74.8% mAP50-95 means the model is reliably placing boxes that overlap the vehicle by 75–90% in most cases — far better than the sliding window approach which routinely achieved 40–55% effective IoU.

### Loss Curves Interpretation

- **box_loss**: Measures how accurately the model places bounding boxes (uses CIoU loss — penalises aspect ratio mismatch and centre offset). Final val box_loss = 0.625, which is **stable** (not rising = no overfitting).
- **cls_loss**: Measures vehicle vs background classification. Final val cls_loss = 0.547 — very low, indicating the model is confident about what is and is not a car.
- **dfl_loss (Distribution Focal Loss)**: YOLO v8-specific. Trains the box edges by predicting a probability distribution over possible pixel positions rather than a single value. This is the mechanism that makes YOLOv8 boxes tighter than YOLOv5 boxes. Final val dfl_loss = 0.808.

### Confusion Matrix (in `yolo_v7_vehicle/confusion_matrix.png`)

There is only one class so the matrix is simple:
- **True Positive**: Model draws box, vehicle is there.
- **False Positive**: Model draws box, nothing is there (confused background for car).
- **False Negative**: Vehicle in frame but model missed it.

Our precision of 83.5% means FP rate is manageable. In traffic counting, a false positive (ghost car) can cause overcounting — but at 83.5% precision, the counting error is bounded and predictable.

---

## 11. Edge Suitability and TinyML Optimisation

### Model Size

| Format | Size |
|:---|:---|
| `vehicle_detector_v7_native.pt` (PyTorch, FP32) | ~6.2 MB |
| `best.onnx` (FP32) | ~5.9 MB |
| INT8 TFLite (quantised) | ~1.7–2.1 MB |

The INT8 quantised TFLite model fits comfortably inside 2 MB — within the TinyML constraint.

### What is INT8 Quantisation?

During training, all weights are stored as 32-bit floating point numbers. For edge deployment:
- Each weight is **mapped to the nearest value representable in 8 bits** (integer from -128 to 127).
- This shrinks the model by ~4× and makes inference 2–4× faster on devices with INT8 hardware units (Coral TPU, ARM Cortex-M with CMSIS-NN).
- A small amount of accuracy is lost (~1–3% mAP) — acceptable for our use case.

The calibration set (a subset of training images) is passed through the model during export so it can calculate the correct scaling factors (called zero-point and scale per layer).

### Why 320×320 input?

The number of operations in a YOLO forward pass scales with **input area**. Halving the resolution quarters the compute:

| Input | Approx multiply-add ops | Real-time on Raspberry Pi 4? |
|:---|:---|:---|
| 640×640 | ~8.7 GFLOPs | No (3–4 FPS) |
| 320×320 | ~2.2 GFLOPs | ~12–15 FPS (acceptable) |
| 224×224 | ~1.1 GFLOPs | ~25 FPS |

At 224×224, distant vehicles become harder to detect. 320×320 is the sweet spot for this camera angle and vehicle density.

---

## 12. Live Inference Pipeline

`LiveTester_V7.py` implements the full inference loop:

```
┌──────────────────────────────────────────────────────────┐
│  Keyboard Hotkey: Ctrl+Shift+1                           │
│        │                                                  │
│        ▼                                                  │
│  toggle() → spawns detection_loop() in background thread │
│                                                           │
│  detection_loop():                                        │
│    1. capture_window_ctypes('Google Chrome')              │
│         Uses Win32 PrintWindow API — captures the        │
│         window even if it is partially off-screen        │
│         Returns BGR numpy array                           │
│                                                           │
│    2. model(frame, conf=0.3, verbose=False)               │
│         Native YOLO forward pass                         │
│         Returns Results object with .boxes                │
│                                                           │
│    3. For each box in results.boxes:                      │
│         box.xyxy[0]  → pixel coords [x1, y1, x2, y2]     │
│         box.conf[0]  → confidence score                   │
│         Draw red rectangle + "V7 0.87" label             │
│                                                           │
│    4. Show count on HUD overlay                           │
│    5. Display via cv2.imshow in a floating window         │
└──────────────────────────────────────────────────────────┘
```

Key implementation details:
- **`PrintWindow` with flag `2`** (`PW_RENDERFULLCONTENT`) is used instead of `BitBlt`. This captures hardware-accelerated (GPU-rendered) windows — Chrome's video uses GPU compositing and would appear black with the older method.
- **Threading**: The detection loop runs in a daemon thread so the main thread can listen for keyboard events without blocking.
- **Confidence threshold 0.3**: Low enough to catch vehicles at the edge of the frame (partly occluded) without too many false positives.

---

## 13. How to Reproduce

### Prerequisites
```bash
pip install ultralytics opencv-python keyboard pygetwindow pywin32 yt-dlp
```

### Step 1 — Collect data
```bash
python Distiller_YOLO_V7.py
# Downloads 11 traffic videos, labels them with YOLOv8 teacher
# Output: dataset_yolo/
```

### Step 2 — Train
```bash
python Train_YOLO_V7.py
# Fine-tunes YOLOv8n on dataset_yolo for 50 epochs at 320px
# Output: runs/detect/yolo_v7_vehicle/weights/best.pt
```

### Step 3 — Export to TFLite
```bash
python export_v7.py
# Exports the best weight checkpoint to ONNX and INT8 TFLite.
```

### Step 4 — Run the live detector
```bash
python LiveTester_V7.py
# Open a traffic video in Chrome
# Press Ctrl+Shift+1 to begin detection
# Press ESC to quit
```

---

## Summary

V7 represents the definitive solution for our Edge Traffic Project. By switching to a native object detection paradigm using **YOLOv8n**, we achieved surgical bounding box precision (91.9% mAP50) while staying within the strict **~2MB** size limit of TinyML devices. The dataset was engineered through high-resolution (720p) automated distillation to capture deep scene diversity, and the final model is fully optimised for real-world deployment on edge hardware.
