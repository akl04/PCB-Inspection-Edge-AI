# ============================================================
# STEP 67D.2: YOLO ARM INFERENCE VERIFICATION
# Raspberry Pi 5
# ============================================================

import os

os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

import json
import time
import hashlib
import platform
import subprocess
import statistics

from pathlib import Path
from datetime import datetime, timezone

import cv2
import torch
import numpy as np

from ultralytics import YOLO

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/home/berry/pcb_inspection_final"
)

MODEL_PATH = Path(
    "/home/berry/pcb_ai/models/best.pt"
)

IMAGE_PATH = Path(
    "/home/berry/pcb_ai/input/pcb_defect_042.jpg"
)

REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANNOTATED_PATH = (
    REPORT_DIR / "yolo_arm_annotated.png"
)

REPORT_PATH = (
    REPORT_DIR / "yolo_arm_test.json"
)

CONFIDENCE_THRESHOLD = 0.25

IMAGE_SIZE = 640

BENCHMARK_RUNS = 5

CLASS_MAPPING = {
    "SH": "Short Circuit",
    "SP": "Spur",
    "SC": "Spurious Copper",
    "OP": "Open Circuit",
    "MB": "Mouse Bite",
    "HB": "Hole Breakout",
    "CS": "Conductor Scratch",
    "CFO": "Conductor Foreign Object",
    "BMFO": "Base Material Foreign Object"
}

print("=" * 65)
print("STEP 67D.2: YOLO ARM INFERENCE VERIFICATION")
print("=" * 65)

# ============================================================
# 2. VERIFY REQUIRED FILES
# ============================================================

if not MODEL_PATH.is_file():

    raise FileNotFoundError(MODEL_PATH)

if not IMAGE_PATH.is_file():

    raise FileNotFoundError(IMAGE_PATH)

print("\nModel:", MODEL_PATH)

print(
    "Model size (bytes):",
    MODEL_PATH.stat().st_size
)

print("Test image:", IMAGE_PATH)

# ============================================================
# 3. CALCULATE MODEL CHECKSUM
# ============================================================

def calculate_sha256(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


model_sha256 = calculate_sha256(
    MODEL_PATH
)

print("\nMODEL SHA-256:")
print(model_sha256)

# ============================================================
# 4. ARM CPU CONFIGURATION
# ============================================================

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

print("\nSYSTEM INFORMATION")

print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("CPU threads:", torch.get_num_threads())
print("Inference device: CPU")

# ============================================================
# 5. LOAD YOLO MODEL
# ============================================================

print("\nLoading YOLO model...")

load_start = time.perf_counter()

model = YOLO(str(MODEL_PATH))

model_load_ms = (
    time.perf_counter() - load_start
) * 1000

print("YOLO model loaded successfully.")

print(
    "Model loading time (ms):",
    round(model_load_ms, 2)
)

print("Model task:", model.task)

if model.task != "detect":

    raise RuntimeError(
        "Expected an object detection model."
    )

# ============================================================
# 6. VERIFY ALL NINE CLASSES
# ============================================================

print("\nYOLO CLASS MAPPING:")

for class_id, label in model.names.items():

    full_name = CLASS_MAPPING.get(
        label,
        "UNKNOWN"
    )

    print(
        f"{class_id}: {label} -> {full_name}"
    )

if set(model.names.values()) != set(CLASS_MAPPING):

    raise RuntimeError(
        "YOLO class labels differ from "
        "the expected nine-class model."
    )

print("\nAll nine expected class labels verified.")

# ============================================================
# 7. LOAD PCB IMAGE
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:

    raise RuntimeError(
        "Failed to load PCB image."
    )

height, width = image.shape[:2]

print("\nIMAGE INFORMATION")

print("Width:", width)
print("Height:", height)

# ============================================================
# 8. WARM UP YOLO
# ============================================================

print("\nWarming up YOLO...", flush=True)

dummy_image = np.zeros(
    (IMAGE_SIZE, IMAGE_SIZE, 3),
    dtype=np.uint8
)

model.predict(
    source=dummy_image,
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE_THRESHOLD,
    device="cpu",
    verbose=False
)

print("Warm-up completed.")

# ============================================================
# 9. RUN FIVE MEASURED INFERENCES
# ============================================================

inference_times = []
prediction_times = []

print("\nRUNNING YOLO ARM BENCHMARK")

for run in range(1, BENCHMARK_RUNS + 1):

    start = time.perf_counter()

    result = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device="cpu",
        verbose=False
    )[0]

    prediction_ms = (
        time.perf_counter() - start
    ) * 1000

    inference_ms = float(
        result.speed["inference"]
    )

    inference_times.append(
        inference_ms
    )

    prediction_times.append(
        prediction_ms
    )

    print(
        f"Run {run}/{BENCHMARK_RUNS} | "
        f"Inference: {inference_ms:.2f} ms | "
        f"Prediction: {prediction_ms:.2f} ms | "
        f"Detections: {len(result.boxes)}",
        flush=True
    )

# ============================================================
# 10. IMAGE-RELATIVE DEFECT LOCATION
# ============================================================

def get_location(x1, y1, x2, y2):

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if cx < width / 3:

        horizontal = "left"

    elif cx < 2 * width / 3:

        horizontal = "center"

    else:

        horizontal = "right"

    if cy < height / 3:

        vertical = "Upper"

    elif cy < 2 * height / 3:

        vertical = "Middle"

    else:

        vertical = "Lower"

    if (
        vertical == "Middle"
        and horizontal == "center"
    ):

        return "Central PCB image region"

    return (
        f"{vertical}-{horizontal} PCB image region"
    )

# ============================================================
# 11. EXTRACT ACTUAL DETECTIONS
# ============================================================

detections = []

print("\nDETECTED PCB DEFECTS")

for index, box in enumerate(result.boxes, 1):

    class_id = int(box.cls.item())

    short_label = model.names[class_id]

    defect = CLASS_MAPPING[short_label]

    confidence = float(box.conf.item())

    x1, y1, x2, y2 = [
        float(v)
        for v in box.xyxy[0].cpu().tolist()
    ]

    detection = {
        "detection_id": index,
        "class_id": class_id,
        "yolo_label": short_label,
        "defect": defect,
        "location": get_location(
            x1, y1, x2, y2
        ),
        "confidence": round(confidence, 4),
        "bounding_box": {
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2)
        }
    }

    detections.append(detection)

    print(
        f"\nDefect {index}: {defect}"
    )

    print(
        "Location:",
        detection["location"]
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print(
        "Bounding box:",
        detection["bounding_box"]
    )

if not detections:

    print(
        "No detections above the selected threshold."
    )

# ============================================================
# 12. SAVE ANNOTATED PCB IMAGE
# ============================================================

annotated = result.plot()

if not cv2.imwrite(
    str(ANNOTATED_PATH),
    annotated
):

    raise IOError(
        "Failed to save annotated image."
    )

# ============================================================
# 13. CALCULATE PERFORMANCE METRICS
# ============================================================

average_inference_ms = statistics.mean(
    inference_times
)

median_inference_ms = statistics.median(
    inference_times
)

average_prediction_ms = statistics.mean(
    prediction_times
)

print("\n" + "=" * 65)
print("YOLO ARM PERFORMANCE SUMMARY")
print("=" * 65)

print("Benchmark runs:", BENCHMARK_RUNS)

print(
    "Average inference (ms):",
    round(average_inference_ms, 2)
)

print(
    "Median inference (ms):",
    round(median_inference_ms, 2)
)

print(
    "Minimum inference (ms):",
    round(min(inference_times), 2)
)

print(
    "Maximum inference (ms):",
    round(max(inference_times), 2)
)

print(
    "Average total prediction (ms):",
    round(average_prediction_ms, 2)
)

print(
    "Total detected defects:",
    len(detections)
)

# ============================================================
# 14. SAVE BENCHMARK REPORT
# ============================================================

report = {
    "timestamp": datetime.now(
        timezone.utc
    ).isoformat(),

    "platform": "Raspberry Pi 5",

    "architecture": platform.machine(),

    "device": "CPU",

    "cpu_threads": torch.get_num_threads(),

    "model_path": str(MODEL_PATH),

    "model_sha256": model_sha256,

    "model_size_bytes": (
        MODEL_PATH.stat().st_size
    ),

    "model_load_ms": round(
        model_load_ms, 2
    ),

    "image": IMAGE_PATH.name,

    "image_width": width,

    "image_height": height,

    "inference_image_size": IMAGE_SIZE,

    "confidence_threshold": (
        CONFIDENCE_THRESHOLD
    ),

    "benchmark_runs": BENCHMARK_RUNS,

    "individual_inference_times_ms": [
        round(t, 2)
        for t in inference_times
    ],

    "individual_prediction_times_ms": [
        round(t, 2)
        for t in prediction_times
    ],

    "average_yolo_inference_ms": round(
        average_inference_ms, 2
    ),

    "median_yolo_inference_ms": round(
        median_inference_ms, 2
    ),

    "minimum_yolo_inference_ms": round(
        min(inference_times), 2
    ),

    "maximum_yolo_inference_ms": round(
        max(inference_times), 2
    ),

    "average_yolo_prediction_ms": round(
        average_prediction_ms, 2
    ),

    "total_detections": len(detections),

    "annotated_image": str(ANNOTATED_PATH),

    "detections": detections
}

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4,
        ensure_ascii=False
    )

# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\nAnnotated image saved to:")
print(ANNOTATED_PATH)

print("\nBenchmark report saved to:")
print(REPORT_PATH)

print("\nCPU TEMPERATURE:")

subprocess.run(
    ["vcgencmd", "measure_temp"],
    check=False
)

print("\n" + "=" * 65)
print("STEP 67D.2 COMPLETED SUCCESSFULLY")
print("=" * 65)

print("YOLO ARM inference: WORKING")
print("Nine-class labels: VERIFIED")
print("Measured benchmark: SAVED")
print("Annotated image: SAVED")
print("READY FOR FINAL DASHBOARD INTEGRATION")
