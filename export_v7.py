"""
export_v7.py
============
V7 Export Script — Converts the trained best.pt model to deployment formats.

Run this after Train_YOLO_V7.py has completed.

Exports:
  1. ONNX (FP32)  — cross-platform, runs on OpenCV DNN, ONNX Runtime
  2. TFLite INT8   — TinyML format for edge hardware (Raspberry Pi, Coral TPU)

The INT8 quantisation shrinks the model from ~6 MB (FP32) to ~1.7 MB,
making inference 2–4× faster on devices with integer acceleration units.
"""

from ultralytics import YOLO
import os

MODEL_PATH = r'runs/detect/yolo_v7_vehicle/weights/best.pt'


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        print("Run Train_YOLO_V7.py first.")
        return

    print("=== export_v7.py — V7 Model Export ===")
    print(f"Source : {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    # ── Export 1: ONNX (FP32) ───────────────────────────────────────────────
    print("\n[1/2] Exporting to ONNX (FP32)...")
    onnx_path = model.export(format='onnx', imgsz=320)
    print(f"  Saved: {onnx_path}")

    # ── Export 2: TFLite INT8 ────────────────────────────────────────────────
    # INT8 quantisation maps each FP32 weight to the nearest int8 value.
    # Requires a calibration dataset (Ultralytics handles this internally
    # using the validation images defined in v7_data.yaml).
    print("\n[2/2] Exporting to TFLite INT8...")
    tflite_path = model.export(format='tflite', imgsz=320, int8=True)
    print(f"  Saved: {tflite_path}")

    print("\n=== Export Complete ===")
    print(f"  ONNX    : {onnx_path}")
    print(f"  TFLite  : {tflite_path}")
    print("\nDeploy vehicle_detector_v7_native.pt for Python inference,")
    print("or the TFLite file for edge/mobile deployment.")


if __name__ == "__main__":
    main()
