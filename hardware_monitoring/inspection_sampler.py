#!/usr/bin/env python3

import logging
import threading
import time
import math

from hardware_monitor import (
    sample_hardware,
    summarize_samples,
)


class InspectionSampler:

    def __init__(self, interval=0.25, sample_function=None):

        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Invalid sampling interval")

        self.interval = interval
        self.sample_function = (
            sample_function or sample_hardware
        )

        self.samples = []
        self.errors = []

        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.thread = None
        self.started = None
        self.finished = None

    def _capture(self):

        try:

            sample = self.sample_function()

            with self.lock:
                self.samples.append(sample)

        except Exception as error:

            with self.lock:
                self.errors.append(str(error))

            logging.warning(
                "Hardware sampling failed: %s",
                error
            )

    def _worker(self):

        while not self.stop_event.is_set():

            cycle_start = time.monotonic()

            self._capture()

            elapsed = time.monotonic() - cycle_start

            delay = max(
                0,
                self.interval - elapsed
            )

            self.stop_event.wait(delay)

    def start(self):

        if self.thread is not None:
            raise RuntimeError("Sampler already started")

        self.started = time.monotonic()

        self.thread = threading.Thread(
            target=self._worker,
            name="pcb-hardware-sampler",
            daemon=True
        )

        self.thread.start()

        return self

    def stop(self):

        if self.thread is None:
            raise RuntimeError("Sampler not started")

        if self.finished is None:

            self.finished = time.monotonic()

            self.stop_event.set()

            self.thread.join(timeout=6)

        return self.result()

    def result(self):

        if self.finished is None:
            raise RuntimeError("Sampler not stopped")

        with self.lock:

            samples = [
                sample for sample in self.samples
                if self.started
                <= sample["monotonic_seconds"]
                <= self.finished
            ]

            errors = list(self.errors)

        samples.sort(
            key=lambda sample:
            sample["monotonic_seconds"]
        )

        duration = self.finished - self.started

        result = {
            "status": "NOT_MEASURED",
            "measurement_scope":
                "monitored_pmic_rails",
            "whole_device_input_power": False,
            "system_wide_measurement": True,
            "requested_interval_seconds":
                self.interval,
            "observation_duration_seconds":
                duration,
            "observation_start_monotonic":
                self.started,
            "observation_end_monotonic":
                self.finished,
            "sample_count": len(samples),
            "sampling_error_count": len(errors),
            "sampling_errors": errors[:10],
            "sampling_thread_stopped":
                not self.thread.is_alive(),
            "sampling_coverage_percent": 0.0,
            "summary": None,
            "samples": samples
        }

        if len(samples) < 2:
            return result

        try:

            summary = summarize_samples(samples)

            sampled_duration = summary[
                "sampling_duration_seconds"
            ]

            coverage = (
                100 * sampled_duration / duration
                if duration > 0 else 0
            )

            gaps = [
                samples[index]["monotonic_seconds"]
                - samples[index - 1]["monotonic_seconds"]
                for index in range(1, len(samples))
            ]

            result.update({
                "status": (
                    "MEASURED"
                    if (
                        coverage >= 90
                        and not errors
                        and max(gaps) <= self.interval * 3
                        and not self.thread.is_alive()
                    )
                    else "PARTIAL_MEASUREMENT"
                ),
                "sampling_coverage_percent":
                    min(100.0, coverage),
                "maximum_sample_gap_seconds":
                    max(gaps),
                "summary": summary
            })

        except Exception as error:

            result["status"] = "MEASUREMENT_FAILED"
            result["summary_error"] = str(error)

        return result


def calculate_performance(
    yolo_inference_ms=None,
    vision_total_ms=None,
    inspection_seconds=None
):

    def positive(value):

        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        )

    return {
        "yolo_inference_fps": (
            1000 / yolo_inference_ms
            if positive(yolo_inference_ms)
            else None
        ),
        "complete_vision_fps": (
            1000 / vision_total_ms
            if positive(vision_total_ms)
            else None
        ),
        "effective_system_fps": (
            1 / inspection_seconds
            if positive(inspection_seconds)
            else None
        ),
        "effective_inspections_per_minute": (
            60 / inspection_seconds
            if positive(inspection_seconds)
            else None
        ),
        "fps_definition":
            "Reciprocal latency, not sustained throughput"
    }
