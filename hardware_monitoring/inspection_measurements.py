#!/usr/bin/env python3

import json
import math
import time
import hashlib

from pathlib import Path

from shared_hardware_monitor import (
    BOOT_ID,
    summarize_window
)

from inspection_sampler import (
    calculate_performance
)


def attach_measurements(
    integrated,
    record,
    finished
):

    record = Path(record)

    performance = calculate_performance(
        yolo_inference_ms=integrated.get(
            "yolo_inference_ms"
        )
    )

    integrated["performance"] = performance

    integrated["hardware_utilization"] = {
        "status": "NOT_MEASURED",
        "summary": None
    }

    timing_path = record.with_suffix(
        ".timing.json"
    )

    # Allow the watcher to finish publishing its
    # supplementary timing sidecar.
    deadline = time.monotonic() + 1

    while (
        not timing_path.is_file()
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)

    if not timing_path.is_file():

        integrated["hardware_utilization"][
            "reason"
        ] = "Vision timing sidecar unavailable"

        return

    timing = json.loads(
        timing_path.read_text()
    )

    actual_hash = hashlib.sha256(
        record.read_bytes()
    ).hexdigest()

    if not (
        timing.get("inspection_id")
        == integrated.get("inspection_id")
        and timing.get("source_record_sha256")
        == integrated.get("source_record_sha256")
        == actual_hash
    ):

        integrated["hardware_utilization"][
            "reason"
        ] = "Inspection identity or checksum mismatch"

        return

    vision_ms = timing.get(
        "yolo_total_processing_ms"
    )

    performance.update(
        calculate_performance(
            yolo_inference_ms=integrated.get(
                "yolo_inference_ms"
            ),
            vision_total_ms=vision_ms
        )
    )

    performance["vision_total_processing_ms"] = (
        vision_ms
    )

    if timing.get("hardware_boot_id") != BOOT_ID:

        integrated["hardware_utilization"][
            "reason"
        ] = "Missing or different hardware boot identity"

        return

    started = timing.get(
        "vision_started_monotonic"
    )

    vision_finished = timing.get(
        "vision_finished_monotonic"
    )

    boundaries = (
        started,
        vision_finished,
        finished
    )

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in boundaries
    ):

        integrated["hardware_utilization"][
            "reason"
        ] = "Invalid inspection boundaries"

        return

    if not started < vision_finished <= finished:

        integrated["hardware_utilization"][
            "reason"
        ] = "Invalid inspection boundary order"

        return

    duration = finished - started

    performance.update(
        calculate_performance(
            yolo_inference_ms=integrated.get(
                "yolo_inference_ms"
            ),
            vision_total_ms=vision_ms,
            inspection_seconds=duration
        )
    )

    performance.update({
        "vision_total_processing_ms": vision_ms,
        "complete_processing_seconds": duration,
        "complete_processing_ms": duration * 1000,
        "measurement_start": "YOLO processing start",
        "measurement_end":
            "Integrated report preparation before monitoring lookup and JSON publication",
        "includes_upload_wait": False,
        "includes_final_report_save": False
    })

    hardware = summarize_window(
        started,
        finished,
        boot_id=BOOT_ID
    )

    hardware.update({
        "inspection_id":
            integrated["inspection_id"],
        "source_record_sha256":
            actual_hash,
        "boot_id": BOOT_ID,
        "measurement_start_monotonic": started,
        "measurement_end_monotonic": finished,
        "measurement_window":
            "Vision start to advisory report preparation",
        "energy_definition":
            "Estimated energy over sampled interval only",
        "attribution":
            "System-wide monitored rails; concurrent workloads included"
    })

    integrated["hardware_utilization"] = hardware


