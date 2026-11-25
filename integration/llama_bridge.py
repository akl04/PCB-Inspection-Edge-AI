#!/usr/bin/env python3

# ============================================================
# AUTOMATIC OFFLINE YOLO-LLAMA INTEGRATION BRIDGE
# Raspberry Pi 5 ARM64
# ============================================================

import json
import time
import os
import signal
import logging
import argparse
import hashlib
import tempfile
import urllib.request
import fcntl

from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT = Path(
    "/home/berry/pcb_inspection_final"
)

CONFIG = PROJECT / "config"

RECORDS = Path(
    "/home/berry/pcb_ai/"
    "auto_inspections/inspection_records"
)

ADVISORIES = PROJECT / "advisories"

LOGS = PROJECT / "logs"

SERVER = "http://127.0.0.1:8091"

MODEL = '/home/berry/pcb_ai/models/llama/pcb_llama_int4_ctx2048.pte'

POLL_INTERVAL = 1.0

RETRY_DELAY = 15.0

REQUEST_TIMEOUT = 300

STOP = False

ADVISORIES.mkdir(
    parents=True,
    exist_ok=True
)

LOGS.mkdir(
    parents=True,
    exist_ok=True
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ============================================================
# 2. LOAD LOCAL CONFIGURATION
# ============================================================

KNOWLEDGE_BASE = json.loads(
    (
        CONFIG / "pcb_knowledge_base.json"
    ).read_text(encoding="utf-8")
)

CLASS_MAPPING = json.loads(
    (
        CONFIG / "yolo_class_mapping.json"
    ).read_text(encoding="utf-8")
)

GUIDANCE_FIELDS = [
    "possible_causes",
    "potential_impact",
    "corrective_action"
]

SEVERITY = "Requires Further Assessment"

if len(CLASS_MAPPING) != 9:
    raise RuntimeError(
        "Expected nine YOLO classes."
    )

if set(KNOWLEDGE_BASE) != set(
    CLASS_MAPPING.values()
):
    raise RuntimeError(
        "Knowledge base and mapping mismatch."
    )

for defect, guidance in KNOWLEDGE_BASE.items():

    for field in GUIDANCE_FIELDS:

        values = guidance.get(field)

        if (
            not isinstance(values, list)
            or not values
            or not all(
                isinstance(v, str) and v.strip()
                for v in values
            )
        ):
            raise RuntimeError(
                f"Invalid guidance: {defect}/{field}"
            )

# ============================================================
# 3. SIX-FIELD JSON SCHEMA
# ============================================================

SCHEMA = {
    "type": "object",

    "properties": {
        "defect": {
            "type": "string"
        },

        "location": {
            "type": "string"
        },

        "severity": {
            "type": "string",
            "enum": [
                SEVERITY
            ]
        },

        "possible_causes": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "potential_impact": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },

        "corrective_action": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },

    "required": [
        "defect",
        "location",
        "severity",
        "possible_causes",
        "potential_impact",
        "corrective_action"
    ],

    "additionalProperties": False
}

SYSTEM_PROMPT = (
    "You are a PCB inspection advisory assistant. "
    "Return exactly the approved_advisory JSON object "
    "provided by the user. "
    "Do not modify, paraphrase, add, or remove guidance. "
    "Do not invent inspection details. "
    "Return only the six required JSON fields."
)

# ============================================================
# 4. UTILITY FUNCTIONS
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def stop_handler(signum, frame):

    global STOP

    STOP = True

    logging.info(
        "Shutdown requested."
    )


signal.signal(
    signal.SIGINT,
    stop_handler
)

signal.signal(
    signal.SIGTERM,
    stop_handler
)


def atomic_json_write(destination, data):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            suffix=".tmp",
            delete=False
        ) as f:

            temporary = Path(f.name)

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

            f.write("\n")

            f.flush()

            os.fsync(f.fileno())

        os.replace(
            temporary,
            destination
        )

    finally:

        if (
            temporary is not None
            and temporary.exists()
        ):

            temporary.unlink()


def check_server():

    with urllib.request.urlopen(
        SERVER + "/health",
        timeout=5
    ) as response:

        health = json.loads(response.read())

    if health.get("status") != "ok":

        raise RuntimeError(
            "ExecuTorch server is not ready."
        )

    if health.get("runtime") != "ExecuTorch":

        raise RuntimeError(
            f"Unexpected runtime: {health.get('runtime')}"
        )

    if health.get("model") != MODEL:

        raise RuntimeError(
            f"Unexpected model: {health.get('model')}"
        )


# ============================================================
# 5. CONVERT EXISTING YOLO LOCATION
# ============================================================

LOCATION_MAPPING = {
    "center region":
        "Central PCB image region",

    "center-left region":
        "Middle-left PCB image region",

    "center-right region":
        "Middle-right PCB image region",

    "top region":
        "Upper-center PCB image region",

    "top-left region":
        "Upper-left PCB image region",

    "top-right region":
        "Upper-right PCB image region",

    "bottom region":
        "Lower-center PCB image region",

    "bottom-left region":
        "Lower-left PCB image region",

    "bottom-right region":
        "Lower-right PCB image region"
}


def normalize_location(location):

    return LOCATION_MAPPING.get(
        location,
        location
    )

# ============================================================
# 6. CREATE REFERENCE ADVISORY
# ============================================================

def create_reference(detection):

    code = detection["defect_class"]

    if code not in CLASS_MAPPING:

        raise ValueError(
            f"Unknown YOLO class: {code}"
        )

    defect = CLASS_MAPPING[code]

    location = normalize_location(
        detection["location"]
    )

    guidance = KNOWLEDGE_BASE[defect]

    return {
        "defect": defect,

        "location": location,

        "severity": SEVERITY,

        "possible_causes":
            guidance["possible_causes"],

        "potential_impact":
            guidance["potential_impact"],

        "corrective_action":
            guidance["corrective_action"]
    }

# ============================================================
# 7. LOCAL LLAMA INFERENCE
# ============================================================

def generate_advisory(detection):

    reference = create_reference(detection)

    # Preserve the existing six-field advisory format.
    final_advisory = json.loads(json.dumps(reference))

    payload = {
        "detection": {
            "defect": reference["defect"],
            "location": reference["location"],
            "confidence": float(detection["confidence"])
        }
    }

    raw_output = None
    generated = None
    selected_option = None
    selected_action = None
    timings = {}
    runtime_metrics = {}
    inference_ms = None
    prompt_ms = None
    generation_ms = None
    request_ms = None
    error = None
    validated = False

    status = "KNOWLEDGE_BASE_FALLBACK"

    logging.info(
        "ExecuTorch advisory for %s | %.2f%%",
        reference["defect"],
        detection["confidence"] * 100
    )

    start = time.perf_counter()

    try:

        request = urllib.request.Request(
            SERVER + "/advisory",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT
        ) as response:

            result = json.loads(response.read())

        request_ms = (
            time.perf_counter() - start
        ) * 1000

        if result.get("runtime") != "ExecuTorch":
            raise ValueError("Unexpected inference runtime")

        if result.get("model") != MODEL:
            raise ValueError("Unexpected Llama model")

        if result.get("defect") != reference["defect"]:
            raise ValueError("Defect class mismatch")

        raw_output = result.get("raw_llama_output")

        timings = result.get("timings", {})

        runtime_metrics = result.get(
            "runtime_metrics", {}
        )

        prompt_ms = timings.get("prompt_ms")

        generation_ms = timings.get("predicted_ms")

        inference_ms = timings.get("inference_ms")

        selected_option = result.get("selected_option")

        selected_action = result.get("selected_action")

        actions = reference["corrective_action"]

        expected_options = {
            "A": actions[0],
            "B": actions[1]
        }

        if (
            result.get("validated") is True
            and result.get("return_code") == 0
            and result.get("prompt_boundary_found") is True
            and result.get("candidate_actions")
                == expected_options
            and selected_option in expected_options
            and selected_action
                == expected_options[selected_option]
        ):

            # Llama determines the priority.
            # The knowledge base supplies the action text.
            # Retain all original corrective actions.

            final_advisory["corrective_action"] = [
                selected_action
            ] + [
                action for action in actions
                if action != selected_action
            ]

            generated = {
                "selected_option": selected_option,
                "selected_action": selected_action
            }

            validated = True

            status = "LLAMA_OUTPUT_VALIDATED"

        else:

            error = (
                result.get("validation_error")
                or "Advisory selection validation failed"
            )

    except Exception as exc:

        error = f"{type(exc).__name__}: {exc}"

        logging.warning(
            "ExecuTorch request or validation failed: %s",
            error
        )

    if request_ms is None:

        request_ms = (
            time.perf_counter() - start
        ) * 1000

    logging.info(
        "Advisory status: %s | Selection: %s | Inference: %s ms",
        status,
        selected_option,
        inference_ms
    )

    return {
        "detection_id": detection["detection_id"],

        "yolo_label": detection["defect_class"],

        "confidence": detection["confidence"],

        "bounding_box_xyxy": detection.get(
            "bounding_box_xyxy"
        ),

        "verification_status": detection.get(
            "verification_status",
            "PENDING_HUMAN_REVIEW"
        ),

        "advisory": final_advisory,

        "generation_status": status,

        # Exact-copy validation is no longer applicable.
        "exact_match": None,

        "selection_validated": validated,

        "validation_error": error,

        "raw_llama_output": raw_output,

        "generated_advisory": generated,

        "runtime": "ExecuTorch",

        "advisory_method": (
            "LLAMA_ACTION_PRIORITIZATION"
            if validated
            else "KNOWLEDGE_BASE_FALLBACK"
        ),

        "selected_option": selected_option,

        "selected_action": (
            selected_action if validated else None
        ),

        "prompt_processing_ms": prompt_ms,

        "token_generation_ms": generation_ms,

        "llama_inference_ms": (
            round(float(inference_ms), 2)
            if inference_ms is not None
            else None
        ),

        "llama_request_ms": round(request_ms, 2),

        "generated_tokens": timings.get("predicted_n"),

        "tokens_per_second": timings.get(
            "predicted_per_second"
        ),

        "model_load_ms": timings.get("model_load_ms"),

        "ttft_ms": timings.get("ttft_ms"),

        "server_timings": timings,

        "runtime_metrics": runtime_metrics
    }


# ============================================================
# 8. PROCESS EXISTING YOLO INSPECTION RECORD
# ============================================================

def process_record(record):

    source_bytes = record.read_bytes()

    source_hash = hashlib.sha256(
        source_bytes
    ).hexdigest()

    data = json.loads(
        source_bytes
    )

    if data.get("status") != "INSPECTION_COMPLETE":

        return

    inspection_id = data["inspection_id"]

    if (
        not inspection_id
        or Path(inspection_id).name != inspection_id
    ):
        raise ValueError(
            "Invalid inspection ID."
        )

    destination = (
        ADVISORIES
        / f"{inspection_id}_integrated.json"
    )

    # Prevent duplicate advisory generation.

    if destination.is_file():

        previous = json.loads(
            destination.read_text(
                encoding="utf-8"
            )
        )

        if (
            previous.get("source_record_sha256")
            == source_hash
            and previous.get("status")
            == "INTEGRATION_COMPLETE"
        ):

            return

    detections = data["detections"]

    if data["total_detections"] != len(detections):

        raise RuntimeError(
            "Detection count mismatch."
        )

    # Validate all class mappings before inference.

    for detection in detections:

        create_reference(detection)

        confidence = float(
            detection["confidence"]
        )

        if not 0 <= confidence <= 1:

            raise ValueError(
                "Invalid detection confidence."
            )

    logging.info(
        "=" * 60
    )

    logging.info(
        "Processing inspection: %s",
        inspection_id
    )

    logging.info(
        "Source record: %s",
        record.name
    )

    logging.info(
        "Total detections: %d",
        len(detections)
    )

    # Avoid processing pending records while
    # the Llama server is unavailable.

    if detections:

        check_server()

    pipeline_start = time.perf_counter()

    advisories = []

    for detection in detections:

        if STOP:

            logging.info(
                "Inspection interrupted; "
                "will retry on next startup."
            )

            return

        advisory = generate_advisory(
            detection
        )

        advisories.append(
            advisory
        )

    pipeline_ms = (
        time.perf_counter() - pipeline_start
    ) * 1000

    # ========================================================
    # 9. CALCULATE ACTUAL TIMING METRICS
    # ========================================================

    measured_times = [
        item["llama_inference_ms"]
        for item in advisories
        if item["llama_inference_ms"]
        is not None
    ]

    validated = sum(
        item["generation_status"]
        == "LLAMA_OUTPUT_VALIDATED"
        for item in advisories
    )

    fallbacks = sum(
        item["generation_status"]
        == "KNOWLEDGE_BASE_FALLBACK"
        for item in advisories
    )

    # Existing watcher records total prediction time,
    # not isolated neural-network inference time.

    yolo_prediction_seconds = data.get(
        "inference_elapsed_seconds"
    )

    yolo_prediction_ms = (
        round(
            float(yolo_prediction_seconds) * 1000,
            2
        )
        if yolo_prediction_seconds is not None
        else None
    )

    yolo_inference_ms = data.get(
        "yolo_inference_ms"
    )

    total_llama_ms = (
        round(sum(measured_times), 2)
        if measured_times
        else None
    )

    average_llama_ms = (
        round(
            sum(measured_times)
            / len(measured_times),
            2
        )
        if measured_times
        else None
    )

    if not detections:

        advisory_status = (
            "NO_DETECTIONS_ABOVE_THRESHOLD"
        )

    elif fallbacks:

        advisory_status = (
            "COMPLETE_WITH_FALLBACK"
        )

    else:

        advisory_status = (
            "ALL_LLAMA_OUTPUTS_VALIDATED"
        )

    # ========================================================
    # 10. SAVE INTEGRATED INSPECTION REPORT
    # ========================================================

    integrated = {
        "inspection_id":
            inspection_id,

        "created_at_utc":
            utc_now(),

        "status":
            "INTEGRATION_COMPLETE",

        "advisory_status":
            advisory_status,

        "platform":
            "Raspberry Pi 5 ARM64",

        "source_record":
            str(record),

        "source_record_sha256":
            source_hash,

        "source_image":
            data["source_image"],

        "annotated_image":
            data["annotated_image"],

        "yolo_model":
            data.get("model_weights"),

        "llama_model":
            MODEL,

        "llama_runtime":
            "ExecuTorch",

        "llama_advisory_method":
            "KNOWLEDGE_GROUNDED_ACTION_PRIORITIZATION",

        "image_width":
            data.get("image_width"),

        "image_height":
            data.get("image_height"),

        "confidence_threshold":
            data.get("confidence_threshold"),

        "total_detections":
            len(detections),

        "total_advisories":
            len(advisories),

        "llama_validated_count":
            validated,

        "knowledge_base_fallback_count":
            fallbacks,

        "llama_timing_sample_count":
            len(measured_times),

        "yolo_inference_ms":
            yolo_inference_ms,

        "yolo_total_prediction_ms":
            yolo_prediction_ms,

        "total_measured_llama_inference_ms":
            total_llama_ms,

        "average_llama_inference_ms":
            average_llama_ms,

        "advisory_pipeline_ms":
            round(pipeline_ms, 2),

        "detections":
            detections,

        "advisories":
            advisories,

        "inspection_notice":
            (
                "AI-assisted preliminary inspection. "
                "Detection confidence is not severity. "
                "Reference guidance requires qualified "
                "human review before corrective action."
            )
    }

    # Supplementary hardware measurements must never
    # prevent publication of an inspection report.
    hardware_finished = time.monotonic()

    try:
        from inspection_measurements import attach_measurements

        attach_measurements(
            integrated,
            record,
            hardware_finished
        )

    except Exception:
        logging.exception(
            "Supplementary measurements unavailable"
        )

    atomic_json_write(
        destination,
        integrated
    )

    logging.info(
        "INSPECTION INTEGRATION COMPLETE"
    )

    logging.info(
        "Advisories delivered: %d",
        len(advisories)
    )

    logging.info(
        "Llama validated: %d",
        validated
    )

    logging.info(
        "Knowledge base fallback: %d",
        fallbacks
    )

    logging.info(
        "YOLO total prediction: %s ms",
        yolo_prediction_ms
    )

    logging.info(
        "Total measured Llama inference: %s ms",
        total_llama_ms
    )

    logging.info(
        "Advisory pipeline: %.2f ms",
        pipeline_ms
    )

    logging.info(
        "Integrated report: %s",
        destination
    )

# ============================================================
# 11. AUTOMATIC RECORD WATCHER
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once",
        action="store_true",
        help="Process existing records once."
    )

    args = parser.parse_args()

    # Prevent two bridge instances from running
    # simultaneously.

    lock_path = LOGS / "llama_bridge.lock"

    with lock_path.open("a+") as lock:

        try:

            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )

        except BlockingIOError:

            raise SystemExit(
                "Another Llama bridge is running."
            )

        if not RECORDS.is_dir():

            raise FileNotFoundError(
                RECORDS
            )

        check_server()

        logging.info(
            "AUTOMATIC OFFLINE LLAMA BRIDGE"
        )

        logging.info(
            "Local Llama server: ONLINE"
        )

        logging.info(
            "Monitoring: %s",
            RECORDS
        )

        logging.info(
            "Output directory: %s",
            ADVISORIES
        )

        logging.info(
            "Polling interval: %.1f seconds",
            POLL_INTERVAL
        )

        retry_after = {}

        while not STOP:

            records = sorted(
                RECORDS.glob(
                    "*_inspection.json"
                )
            )

            failures = 0

            for record in records:

                if STOP:
                    break

                if time.monotonic() < retry_after.get(
                    record,
                    0
                ):

                    continue

                try:

                    process_record(
                        record
                    )

                    retry_after.pop(
                        record,
                        None
                    )

                except Exception:

                    failures += 1

                    logging.exception(
                        "Failed to process %s",
                        record.name
                    )

                    retry_after[record] = (
                        time.monotonic()
                        + RETRY_DELAY
                    )

            if args.once:

                if failures or STOP:

                    raise SystemExit(1)

                break

            time.sleep(
                POLL_INTERVAL
            )

        logging.info(
            "Llama bridge stopped."
        )


if __name__ == "__main__":

    main()
