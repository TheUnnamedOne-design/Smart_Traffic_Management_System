"""
Train_YOLO_V7.py
================
V7 Training Script — Fine-tune YOLOv8n on the collected traffic dataset.

Reads the dataset produced by Distiller_YOLO_V7.py (stored in dataset_yolo/)
and fine-tunes a YOLOv8n model for 50 epochs at 320×320 resolution.

Output:
    runs/detect/yolo_v7_vehicle/
        weights/best.pt      ← Best checkpoint (used for deployment)
        weights/last.pt      ← Final epoch checkpoint
        results.csv          ← Per-epoch metrics (mAP, Precision, Recall, Loss)
        results.png          ← Loss/metric curves plot
        confusion_matrix.png
        val_batch*.jpg       ← Validation predictions visualised
"""

from ultralytics import YOLO


def main():
    # ── Start from COCO pretrained YOLOv8n ──────────────────────────────────
    # Using pretrained weights means the backbone already knows how to detect
    # edges, shapes, and textures. We only need to specialise the detection
    # head for our single "vehicle" class and camera angle.
    model = YOLO('yolov8n.pt')

    print("=== Train_YOLO_V7.py — V7 Fine-Tuning ===")
    print("Dataset : v7_data.yaml")
    print("Epochs  : 50")
    print("imgsz   : 320")
    print()

    results = model.train(
        data='v7_data.yaml',   # Points to dataset_yolo/ train+val dirs
        epochs=50,             # Full convergence; loss plateaus ~epoch 38
        imgsz=320,             # TinyML constraint: 320px reduces FLOPs 4× vs 640px
        batch=16,              # Fits CPU comfortably; increase for GPU
        name='yolo_v7_vehicle',# Output folder: runs/detect/yolo_v7_vehicle/
        device='cpu',          # CPU training; change to 0 for CUDA GPU
        # Augmentations applied automatically by Ultralytics:
        #   Mosaic, RandomFlip, HSV jitter, Scale/crop
    )

    print("\n=== Training Complete ===")
    print(f"Best weights : runs/detect/yolo_v7_vehicle/weights/best.pt")
    print(f"Metrics CSV  : runs/detect/yolo_v7_vehicle/results.csv")
    print(f"\nFinal mAP50   : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"Final mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"Final Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"Final Recall   : {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
    print("\nNext step: python export_v7.py")


if __name__ == "__main__":
    main()
