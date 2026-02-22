# V7 TinyML Vehicle Detector - Implementation Flow

## Diagram 1: Overall Pipeline (High-Level)

```mermaid
flowchart LR
    Start([Start]) --> Phase1[1. Data Collection<br/>Distiller_YOLO_V7.py]
    Phase1 --> Phase2[2. Model Training<br/>Train_YOLO_V7.py]
    Phase2 --> Phase3[3. Model Export<br/>export_v7.py]
    Phase3 --> Phase4[4. Live Inference<br/>LiveTester_V7.py]
    Phase4 --> End([Deployed])
    
    style Phase1 fill:#e1f5ff,stroke:#01579b,stroke-width:3px,color:#000
    style Phase2 fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
    style Phase3 fill:#f3e5f5,stroke:#4a148c,stroke-width:3px,color:#000
    style Phase4 fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px,color:#000
```

---

## Diagram 2: Phase 1 - Data Collection

```mermaid
flowchart TD
    Start([Distiller_YOLO_V7.py]) --> Download[Download 11 YouTube Videos<br/>720p Traffic Footage]
    Download --> Teacher[Load YOLOv8n Teacher Model<br/>COCO Pretrained]
    Teacher --> Sample[Sample Frames<br/>Every 30 frames]
    Sample --> Label[Auto-Label Vehicles<br/>Confidence ≥ 0.4]
    Label --> Convert[Convert to YOLO Format<br/>Normalized Coordinates]
    Convert --> Split[Split 80/20<br/>Train/Validation]
    Split --> Save[Save Dataset<br/>dataset_yolo/]
    Save --> YAML[Generate v7_data.yaml]
    YAML --> End([Dataset Ready])
    
    style Start fill:#e1f5ff,stroke:#01579b,stroke-width:2px,color:#000
    style End fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
```

---

## Diagram 3: Phase 2 - Model Training

```mermaid
flowchart TD
    Start([Train_YOLO_V7.py]) --> Load[Load Pretrained YOLOv8n<br/>COCO Weights]
    Load --> Config[Configure Training<br/>50 Epochs, 320x320, Batch=16]
    Config --> Train[Fine-tune on Traffic Dataset<br/>With Data Augmentation]
    Train --> Loop{Epoch Loop<br/>50 Iterations}
    Loop --> Validate[Validate on Val Set<br/>Calculate mAP]
    Validate --> Save[Save Best Checkpoint<br/>best.pt]
    Save --> Loop
    Loop --> Metrics[Generate Metrics<br/>results.csv]
    Metrics --> End([Model Trained])
    
    style Start fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style End fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
```

---

## Diagram 4: Phase 3 - Model Export

```mermaid
flowchart TD
    Start([export_v7.py]) --> Load[Load Best Checkpoint<br/>best.pt]
    Load --> ONNX[Export to ONNX<br/>FP32 Format]
    ONNX --> TFLite[Export to TFLite<br/>INT8 Quantized ~1.7MB]
    TFLite --> End([Deployment Ready<br/>best.onnx + best.tflite])
    
    style Start fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    style End fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
```

---

## Diagram 5: Phase 4 - Live Inference

```mermaid
flowchart TD
    Start([LiveTester_V7.py]) --> LoadModel[Load Trained Model<br/>vehicle_detector_v7_native.pt]
    LoadModel --> Capture[Capture Browser Window<br/>Chrome/YouTube]
    Capture --> Inference[Run YOLO Inference<br/>Confidence = 0.3]
    Inference --> Draw[Draw Bounding Boxes<br/>Show Vehicle Count]
    Draw --> Display[Display Live Output]
    Display --> Capture
    
    style Start fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    style Display fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
```
