import math
import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json
import math
import html
import subprocess
import urllib.request
import inspect
import re
import uuid
import shutil
import tempfile
import warnings
import sqlite3
import hashlib
import threading

from datetime import datetime, timezone

from pathlib import Path

import gradio as gr
import pandas as pd

from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path("/home/berry/pcb_inspection_final")

INPUT_DIR = Path("/home/berry/pcb_ai/input")

YOLO_OUTPUT = Path("/home/berry/pcb_ai/auto_inspections")

RECORDS_DIR = YOLO_OUTPUT / "inspection_records"

ANNOTATED_DIR = YOLO_OUTPUT / "annotated_images"

ADVISORIES_DIR = PROJECT / "advisories"

LLAMA_SERVER = "http://127.0.0.1:8091"

REFRESH_SECONDS = 3

DELETED_RECORDS_DIR = PROJECT / "deleted_inspections"

WATCHER_DATABASE = YOLO_OUTPUT / ".watcher_state.sqlite3"

HISTORY_LOCK = threading.RLock()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

MAX_IMAGE_PIXELS = 40_000_000

SUPPORTED_IMAGE_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "WEBP": ".webp"
}

CLASS_MAPPING = json.loads(
    (PROJECT / "config/yolo_class_mapping.json").read_text()
)

TABLE_COLUMNS = [
    "ID",
    "Defect",
    "Location",
    "Confidence (%)",
    "Llama inference (ms)",
    "Advisory source"
]

# ============================================================
# DASHBOARD DESIGN
# ============================================================

CSS = """
.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
}

#main-header {
    background: linear-gradient(
        120deg, #0f172a, #193454, #23466a
    );
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 12px;
}


#main-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 24px;
}

.header-description {
    flex: 1 1 480px;
}

.header-performance {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
}

.header-performance-card {
    background: rgba(5, 18, 35, 0.55);
    border: 1px solid rgba(147, 197, 253, 0.25);
    border-radius: 12px;
    padding: 18px;
    min-width: 155px;
    text-align: center;
}

.header-performance-label {
    color: #cbd5e1;
    font-size: 12px;
    font-weight: 600;
}

.header-fps {
    color: #60a5fa;
    font-size: 32px;
    font-weight: 750;
    margin: 8px 0;
}

.header-total {
    color: #4ade80;
    font-size: 32px;
    font-weight: 750;
    margin: 8px 0;
}

.header-performance-detail {
    color: #94a3b8;
    font-size: 11px;
}

#main-header h1 {
    color: white !important;
    font-size: 30px;
    margin: 8px 0;
}

#main-header p {
    color: #cbd5e1;
    margin: 5px 0;
}

.eyebrow {
    color: #93c5fd;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 10px 0;
}

.metric-card {
    background: var(--background-fill-secondary, #f8fafc);
    border: 1px solid var(--border-color-primary, #dbe3ed);
    border-radius: 12px;
    padding: 18px;
}

.metric-label {
    color: var(--body-text-color-subdued, #64748b);
    font-size: 13px;
    margin-bottom: 8px;
}

.metric-value {
    color: var(--body-text-color, #163354);
    font-size: 25px;
    font-weight: 750;
}

.metric-detail {
    color: var(--body-text-color-subdued, #64748b);
    font-size: 11px;
    margin-top: 5px;
}

.status-panel {
    border: 1px solid var(--border-color-primary, #dbe3ed);
    border-radius: 10px;
    padding: 14px;
}

.status-online {
    color: #16a34a;
    font-weight: 650;
}

.status-warning {
    color: #d97706;
    font-weight: 650;
}

.advisory-panel {
    border: 1px solid var(--border-color-primary, #dbe3ed);
    border-radius: 12px;
    padding: 20px;
}

.advisory-section {
    margin: 15px 0;
}

.advisory-label {
    font-size: 12px;
    font-weight: 700;
    color: var(--body-text-color-subdued, #64748b);
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.advisory-value {
    font-size: 16px;
    margin-top: 6px;
}

.advisory-section ul {
    margin: 8px 0;
    padding-left: 22px;
}

.advisory-section li {
    margin: 6px 0;
}

.severity-value {
    color: #d97706;
    font-weight: 650;
}

@media (max-width: 800px) {
    .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .metric-value {
        font-size: 20px;
    }

    #main-header h1 {
        font-size: 23px;
    }
}
"""

# ============================================================
# UTILITIES
# ============================================================

def esc(value):
    return html.escape(str(value))


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def fmt_ms(value):

    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ):
        return f"{value:,.2f} ms"

    return "Not measured"


def confidence_pct(value):

    try:
        return round(float(value) * 100, 2)

    except (TypeError, ValueError):
        return None


def empty_table():
    return pd.DataFrame(columns=TABLE_COLUMNS)


def source_label(status):

    return {
        "LLAMA_OUTPUT_VALIDATED": "Llama validated",
        "KNOWLEDGE_BASE_FALLBACK": "Knowledge base fallback"
    }.get(status, status or "Pending")


def safe_path(value, directory):

    if not value:
        return None

    try:

        path = Path(value).resolve()

        if not path.is_relative_to(directory.resolve()):
            return None

        if not path.is_file():
            return None

        return path

    except (OSError, ValueError, TypeError):
        return None


def load_image(value, directory):

    path = safe_path(value, directory)

    if path is None:
        return None

    try:

        with Image.open(path) as image:

            image = image.convert("RGB")

            image.thumbnail((1400, 1400))

            return image.copy()

    except (OSError, ValueError):
        return None


# ============================================================
# AI SERVICE MONITORING
# ============================================================


# ============================================================
# PCB IMAGE SUBMISSION
# ============================================================

def save_uploaded_pcb(uploaded_file):

    if not uploaded_file:
        raise ValueError(
            "Please select a PCB image first."
        )

    source_path = Path(str(uploaded_file))

    if not source_path.is_file():
        raise ValueError(
            "The selected image could not be accessed."
        )

    size = source_path.stat().st_size

    if size <= 0:
        raise ValueError(
            "The selected file is empty."
        )

    if size > MAX_UPLOAD_BYTES:
        raise ValueError(
            "The selected image exceeds the 25 MB limit."
        )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "error",
            Image.DecompressionBombWarning
        )

        with Image.open(source_path) as image:

            image_format = image.format

            width, height = image.size

            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError(
                    "Unsupported image format. "
                    "Use JPG, JPEG, PNG, BMP or WEBP."
                )

            if (
                width <= 0
                or height <= 0
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise ValueError(
                    "Invalid image dimensions or "
                    "image exceeds 40 megapixels."
                )

            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(
                    "Animated or multi-frame images "
                    "are not supported."
                )

            image.verify()

        with Image.open(source_path) as image:
            image.load()

    suffix = SUPPORTED_IMAGE_FORMATS[image_format]

    safe_stem = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        source_path.stem
    ).strip("_-")[:60]

    if not safe_stem:
        safe_stem = "pcb"

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    unique_id = uuid.uuid4().hex[:10]

    filename = (
        f"{safe_stem}_{stamp}_{unique_id}{suffix}"
    )

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = INPUT_DIR / filename

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".pcb_upload_",
            suffix=".tmp",
            dir=INPUT_DIR,
            delete=False
        ) as temporary:

            temporary_path = Path(temporary.name)

            with source_path.open("rb") as original:

                shutil.copyfileobj(
                    original,
                    temporary
                )

            temporary.flush()

            os.fsync(temporary.fileno())

        os.replace(
            temporary_path,
            destination
        )

    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

    return {
        "filename": filename,
        "source_image": str(destination),
        "width": width,
        "height": height,
        "size_bytes": size,
        "submitted_at_utc": (
            datetime.now(timezone.utc).isoformat()
        )
    }


def submit_pcb_image(uploaded_file, previous_submission):

    try:

        submission = save_uploaded_pcb(
            uploaded_file
        )

    except Exception as error:

        return (
            previous_submission,
            (
                "**Submission failed.** "
                + esc(str(error))
            ),
            gr.skip(),
            gr.skip()
        )

    message = (
        "**Image submitted successfully.**\n\n"
        f"File: `{submission['filename']}`\n\n"
        f"Dimensions: {submission['width']} "
        f"x {submission['height']} pixels\n\n"
        "Waiting for automatic YOLO inspection."
    )

    return (
        submission,
        message,
        True,
        None
    )


def get_submission_status(submission):

    if not submission:

        return (
            "Ready for image submission. "
            "Select a raw PCB image to begin."
        )

    filename = submission["filename"]

    expected_path = Path(
        submission["source_image"]
    )

    inspections = get_inspections()

    for inspection_id, item in inspections.items():

        record = item["record"]

        source_image = record.get("source_image")

        if not source_image:
            continue

        if Path(source_image) != expected_path:
            continue

        count = record.get(
            "total_detections", 0
        )

        report = item["report"]

        if report is None:

            return (
                "**YOLO inspection complete.**\n\n"
                f"File: `{filename}`\n\n"
                f"Detected defects: **{count}**\n\n"
                "Waiting for automatic Llama "
                "advisory generation."
            )

        if count == 0:

            return (
                "**Inspection complete.**\n\n"
                f"File: `{filename}`\n\n"
                "No defects detected above the "
                "configured confidence threshold. "
                "This does not certify a defect-free PCB."
            )

        validated = report.get(
            "llama_validated_count", 0
        )

        fallbacks = report.get(
            "knowledge_base_fallback_count", 0
        )

        return (
            "**Inspection and advisory generation "
            "complete.**\n\n"
            f"File: `{filename}`\n\n"
            f"Detected defects: **{count}** | "
            f"Llama validated: **{validated}** | "
            f"Knowledge base fallbacks: **{fallbacks}**\n\n"
            "Results are available below or in "
            "the inspection history."
        )

    if not expected_path.is_file():

        return (
            "**Submitted image is no longer available.** "
            "Please submit another image."
        )

    return (
        "**Image submitted successfully.**\n\n"
        f"File: `{filename}`\n\n"
        "Queued for automatic YOLO inspection. "
        "Please wait for the results."
    )



def get_system_status():

    try:

        result = subprocess.run(
            [
                "systemctl",
                "is-active",
                "pcb-yolo-watcher.service"
            ],
            capture_output=True,
            text=True,
            timeout=3
        )

        yolo_online = result.stdout.strip() == "active"

    except Exception:
        yolo_online = False

    try:

        with urllib.request.urlopen(
            LLAMA_SERVER + "/health",
            timeout=2
        ) as response:

            llama_online = (
                json.load(response).get("status") == "ok"
            )

    except Exception:
        llama_online = False

    try:

        result = subprocess.run(
            [
                "pgrep",
                "-f",
                "python3.*[/]llama_bridge.py"
            ],
            capture_output=True,
            text=True,
            timeout=3
        )

        bridge_online = result.returncode == 0

    except Exception:
        bridge_online = False

    def status(name, online):

        style = (
            "status-online"
            if online
            else "status-warning"
        )

        label = (
            "Running"
            if online
            else "Unavailable"
        )

        return (
            f'<span class="{style}">'
            f'● {name}: {label}</span>'
        )

    return (
        '<div class="status-panel">'
        + status("YOLO", yolo_online)
        + " &nbsp; | &nbsp; "
        + status("Llama server", llama_online)
        + " &nbsp; | &nbsp; "
        + status("Llama bridge", bridge_online)
        + "</div>"
    )


# ============================================================
# READ EXISTING INSPECTION RECORDS
# ============================================================

def get_inspections():

    inspections = {}

    for path in RECORDS_DIR.glob("*_inspection.json"):

        try:

            record = read_json(path)

            if record.get("status") != "INSPECTION_COMPLETE":
                continue

            inspection_id = record["inspection_id"]

            if Path(inspection_id).name != inspection_id:
                continue

            report_path = ADVISORIES_DIR / (
                inspection_id + "_integrated.json"
            )

            report = None

            if report_path.is_file():

                candidate = read_json(report_path)

                if (
                    candidate.get("status")
                    == "INTEGRATION_COMPLETE"
                    and candidate.get("inspection_id")
                    == inspection_id
                ):
                    report = candidate

            inspections[inspection_id] = {
                "record": record,
                "record_path": str(path),
                "report": report,
                "report_path": (
                    str(report_path)
                    if report is not None
                    else None
                )
            }

        except (OSError, ValueError, KeyError, TypeError):
            continue

    return dict(
        sorted(
            inspections.items(),
            key=lambda item: item[0],
            reverse=True
        )
    )


def inspection_choices(inspections):

    choices = []

    for inspection_id, item in inspections.items():

        record = item["record"]

        filename = Path(
            record.get("source_image", "")
        ).name

        state = (
            "Complete"
            if item["report"] is not None
            else "Advisory pending"
        )

        label = (
            f"{filename} | "
            f"{record.get('total_detections', 0)} defects | "
            f"{state}"
        )

        choices.append((label, inspection_id))

    return choices


# ============================================================
# PERFORMANCE METRIC CARDS
# ============================================================

def metric_card(label, value, detail):

    return f"""
    <div class="metric-card">
        <div class="metric-label">{esc(label)}</div>
        <div class="metric-value">{esc(value)}</div>
        <div class="metric-detail">{esc(detail)}</div>
    </div>
    """


def render_metrics(record=None, report=None):

    record = record or {}

    total = record.get("total_detections", 0)

    if report is None:

        average = "Pending"
        total_llama = "Pending"
        validated = "Pending"
        fallbacks = "Pending"

    else:

        average = fmt_ms(
            report.get("average_llama_inference_ms")
        )

        samples = report.get(
            "llama_timing_sample_count", 0
        )

        total_llama = (
            fmt_ms(
                report.get(
                    "total_measured_llama_inference_ms"
                )
            )
            if samples
            else "Not measured"
        )

        validated = (
            f"{report.get('llama_validated_count', 0)}"
            f" / {total}"
        )

        fallbacks = str(
            report.get(
                "knowledge_base_fallback_count", 0
            )
        )

    cards = [

        metric_card(
            "Detected defects",
            total,
            "Above YOLO confidence threshold"
        ),

        metric_card(
            "YOLO inference",
            fmt_ms(record.get("yolo_inference_ms")),
            "Measured model inference per image"
        ),

        metric_card(
            "Average Llama inference",
            average,
            "Per measured defect advisory"
        ),

        metric_card(
            "Total Llama inference",
            total_llama,
            "Sum of measured inference times"
        ),

        metric_card(
            "Validated Llama advisories",
            validated,
            "Exact knowledge base agreement"
        ),

        metric_card(
            "Knowledge base fallbacks",
            fallbacks,
            "Fallback guidance when required"
        )
    ]

    return (
        '<div class="metric-grid">'
        + "".join(cards)
        + "</div>"
    )


# ============================================================
# DETECTED DEFECT TABLE
# ============================================================

def render_defect_table(record, report):

    advisory_map = {}

    if report is not None:

        advisory_map = {
            str(item["detection_id"]): item
            for item in report.get("advisories", [])
        }

    rows = []

    for detection in record.get("detections", []):

        detection_id = str(
            detection["detection_id"]
        )

        item = advisory_map.get(detection_id, {})

        advisory = item.get("advisory", {})

        code = detection.get("defect_class", "")

        defect = advisory.get(
            "defect",
            CLASS_MAPPING.get(code, code)
        )

        location = advisory.get(
            "location",
            detection.get("location", "")
        )

        inference_ms = item.get("llama_inference_ms")

        rows.append([
            detection_id,
            defect,
            location,
            confidence_pct(detection.get("confidence")),
            (
                round(inference_ms, 2)
                if isinstance(inference_ms, (int, float))
                else None
            ),
            source_label(item.get("generation_status"))
        ])

    return pd.DataFrame(
        rows,
        columns=TABLE_COLUMNS
    )


# ============================================================
# SIX-FIELD ENGINEERING ADVISORY
# ============================================================

def render_advisory(item):

    if not item:

        return (
            '<div class="advisory-panel">'
            'No completed defect advisory selected.'
            '</div>',
            {},
            ""
        )

    advisory = item.get("advisory", {})

    sections = [
        ("Defect", "defect"),
        ("Location", "location"),
        ("Severity", "severity"),
        ("Possible causes", "possible_causes"),
        ("Potential impact", "potential_impact"),
        ("Corrective action", "corrective_action")
    ]

    output = ['<div class="advisory-panel">']

    for label, key in sections:

        value = advisory.get(key, "")

        output.append(
            '<div class="advisory-section">'
            f'<div class="advisory-label">{esc(label)}</div>'
        )

        if isinstance(value, list):

            output.append("<ul>")

            for entry in value:

                output.append(
                    f"<li>{esc(entry)}</li>"
                )

            output.append("</ul>")

        else:

            style = (
                "advisory-value severity-value"
                if key == "severity"
                else "advisory-value"
            )

            output.append(
                f'<div class="{style}">'
                f'{esc(value)}</div>'
            )

        output.append("</div>")

    output.append("</div>")

    confidence = confidence_pct(
        item.get("confidence")
    )

    metadata = (
        f"**YOLO confidence:** {confidence}%\n\n"
        f"**Advisory source:** "
        f"{source_label(item.get('generation_status'))}\n\n"
        f"**Llama inference:** "
        f"{fmt_ms(item.get('llama_inference_ms'))}\n\n"
        f"**Llama request latency:** "
        f"{fmt_ms(item.get('llama_request_ms'))}"
    )

    if item.get("validation_error"):

        metadata += (
            "\n\n**Validation note:** "
            + esc(item["validation_error"])
        )

    return "".join(output), advisory, metadata


def select_advisory(report, detection_id):

    if report is None or detection_id is None:
        return render_advisory(None)

    for item in report.get("advisories", []):

        if str(item["detection_id"]) == str(detection_id):
            return render_advisory(item)

    return render_advisory(None)


# ============================================================
# DETAILED INFERENCE TIMING
# ============================================================

def timing_details(record, report):

    elapsed = record.get("inference_elapsed_seconds")

    total_prediction = (
        elapsed * 1000
        if isinstance(elapsed, (int, float))
        else None
    )

    lines = [
        "### Detailed runtime measurements",
        "",
        f"**YOLO preprocessing:** "
        f"{fmt_ms(record.get('yolo_preprocess_ms'))}",
        "",
        f"**YOLO model inference:** "
        f"{fmt_ms(record.get('yolo_inference_ms'))}",
        "",
        f"**YOLO postprocessing:** "
        f"{fmt_ms(record.get('yolo_postprocess_ms'))}",
        "",
        f"**YOLO total prediction:** "
        f"{fmt_ms(total_prediction)}"
    ]

    if report is not None:

        lines.extend([
            "",
            f"**Llama advisory pipeline:** "
            f"{fmt_ms(report.get('advisory_pipeline_ms'))}",
            "",
            f"**Llama timing samples:** "
            f"{report.get('llama_timing_sample_count', 0)}"
            f" / {record.get('total_detections', 0)}"
        ])

    lines.extend([
        "",
        "Historical records without isolated YOLO "
        "measurements are displayed as Not measured.",
        "",
        "YOLO inference is measured per image. "
        "Llama inference includes prompt processing "
        "and token generation per advisory."
    ])

    return "\n".join(lines)


# ============================================================
# LOAD INSPECTION RESULTS
# ============================================================

def load_inspection(
    inspection_id,
    previous_defect=None,
    inspections=None
):

    if inspections is None:
        inspections = get_inspections()

    item = inspections.get(inspection_id)

    if item is None:

        return (
            render_metrics(),
            None,
            None,
            empty_table(),
            gr.update(choices=[], value=None),
            None,
            *render_advisory(None),
            None,
            None,
            "Waiting for automatic YOLO inspection.",
            ""
        )

    record = item["record"]

    report = item["report"]

    original = load_image(
        record.get("source_image"),
        INPUT_DIR
    )

    annotated = load_image(
        record.get("annotated_image"),
        ANNOTATED_DIR
    )

    choices = []

    if report is not None:

        for advisory_item in report.get("advisories", []):

            detection_id = str(
                advisory_item["detection_id"]
            )

            advisory = advisory_item["advisory"]

            choices.append((
                f"Defect {detection_id} — "
                f"{advisory.get('defect', '')}",
                detection_id
            ))

    valid_ids = [value for _, value in choices]

    selected = (
        str(previous_defect)
        if str(previous_defect) in valid_ids
        else valid_ids[0] if valid_ids else None
    )

    advisory_html, advisory_json, metadata = (
        select_advisory(report, selected)
    )

    if report is None:

        notice = (
            "**YOLO inspection complete.** "
            "Waiting for automatic Llama advisory generation."
        )

    elif record.get("total_detections", 0) == 0:

        notice = (
            "**Inspection complete.** "
            "No detections above the configured threshold. "
            "This does not certify a defect-free PCB."
        )

    else:

        notice = (
            "**Automatic inspection complete.** "
            "Advisories are preliminary and require "
            "human engineering assessment. "
            "Detection confidence is not defect severity."
        )

    return (
        render_metrics(record, report),
        original,
        annotated,
        render_defect_table(record, report),
        gr.update(choices=choices, value=selected),
        report,
        advisory_html,
        advisory_json,
        metadata,
        item["record_path"],
        item["report_path"],
        notice,
        timing_details(record, report)
    )


# ============================================================
# AUTOMATIC DASHBOARD REFRESH
# ============================================================

def refresh_dashboard(
    current_inspection,
    follow_latest,
    current_defect,
    previous_signature
):

    inspections = get_inspections()

    choices = inspection_choices(inspections)

    if not choices:

        selected = None

    elif follow_latest or current_inspection not in inspections:

        selected = choices[0][1]

    else:

        selected = current_inspection

    item = inspections.get(selected)

    signature = None

    if item is not None:

        try:

            signature = [
                selected,
                item["record_path"],
                item["report_path"],
                Path(
                    item["record_path"]
                ).stat().st_mtime_ns,
                (
                    Path(
                        item["report_path"]
                    ).stat().st_mtime_ns
                    if item["report_path"]
                    else None
                )
            ]

        except OSError:
            signature = None

    selector_update = gr.update(
        choices=choices,
        value=selected
    )

    status = get_system_status()

    if (
        previous_signature is not None
        and previous_signature == signature
        and selected == current_inspection
    ):

        return (
            selector_update,
            status,
            signature,
            *[gr.skip() for _ in range(16)]
        )

    return (
        selector_update,
        status,
        signature,
        *load_inspection(
            selected,
            (
                current_defect
                if selected == current_inspection
                else None
            ),
            inspections
        )
    )


def change_inspection(inspection_id):

    return (
        False,
        None,
        *load_inspection(inspection_id)
    )


# ============================================================
# FINAL HISTORY MANAGEMENT
# ============================================================

# ============================================================
# FINAL DASHBOARD HISTORY MANAGEMENT
# ============================================================

def blank_view():

    return load_inspection(None, inspections={})


def history_update(selected=None):

    inspections = get_inspections()

    if selected not in inspections:
        selected = None

    return gr.update(
        choices=inspection_choices(inspections),
        value=selected
    )


def resolve_active_inspection(view_state, inspections):

    if not view_state:
        return None

    mode = view_state.get("mode")

    if mode == "history":

        inspection_id = view_state.get("inspection_id")

        if inspection_id in inspections:
            return inspection_id

        return None

    if mode == "submission":

        expected = view_state.get("source_image")

        for inspection_id, item in inspections.items():

            if item["record"].get("source_image") == expected:
                return inspection_id

    return None


def final_refresh(
    selected_history,
    view_state,
    current_defect,
    previous_signature,
    submission
):

    with HISTORY_LOCK:

        inspections = get_inspections()

        selected = resolve_active_inspection(
            view_state,
            inspections
        )

        item = inspections.get(selected)

        signature = None

        if item is not None:

            try:

                signature = [
                    selected,
                    Path(
                        item["record_path"]
                    ).stat().st_mtime_ns,
                    (
                        Path(
                            item["report_path"]
                        ).stat().st_mtime_ns
                        if item["report_path"]
                        else None
                    )
                ]

            except OSError:
                signature = None

        if signature is not None:

            timing_path = Path(
                item["record_path"]
            ).with_suffix(".timing.json")

            try:

                signature.append(
                    timing_path.stat().st_mtime_ns
                )

            except OSError:

                signature.append(None)

        if selected_history not in inspections:
            selected_history = None

        history = gr.update(
            choices=inspection_choices(inspections),
            value=selected_history
        )

        status = get_system_status()

        submission_message = get_submission_status(
            submission
        )

        if (
            signature is not None
            and signature == previous_signature
        ):

            results = [
                gr.skip() for _ in range(16)
            ]

        else:

            results = load_inspection(
                selected,
                current_defect,
                inspections
            )

        return (
            history,
            status,
            signature,
            submission_message,
            *results
        )


def reset_dashboard():

    with HISTORY_LOCK:

        return (
            history_update(),
            None,
            None,
            None,
            None,
            "Ready for image submission. "
            "Select a raw PCB image to begin.",
            "",
            None,
            gr.update(visible=False),
            *blank_view()
        )


def submit_new_inspection(uploaded_file):

    with HISTORY_LOCK:

        try:

            submission = save_uploaded_pcb(
                uploaded_file
            )

        except Exception as error:

            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "**Submission failed:** "
                + esc(str(error)),
                gr.skip(),
                None,
                gr.update(visible=False),
                *[gr.skip() for _ in range(16)]
            )

        view = {
            "mode": "submission",
            "source_image": submission["source_image"]
        }

        return (
            view,
            submission,
            None,
            None,
            get_submission_status(submission),
            "",
            None,
            gr.update(visible=False),
            *blank_view()
        )


def load_previous_record(inspection_id):

    with HISTORY_LOCK:

        inspections = get_inspections()

        if inspection_id not in inspections:

            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "Please select an existing inspection.",
                gr.skip(),
                None,
                gr.update(visible=False),
                *[gr.skip() for _ in range(16)]
            )

        view = {
            "mode": "history",
            "inspection_id": inspection_id
        }

        return (
            view,
            None,
            None,
            "**Previous inspection loaded.**",
            "Viewing a saved inspection.",
            None,
            gr.update(visible=False),
            *load_inspection(
                inspection_id,
                inspections=inspections
            )
        )


def request_record_delete(inspection_id):

    with HISTORY_LOCK:

        inspections = get_inspections()

        item = inspections.get(inspection_id)

        if item is None:

            return (
                None,
                gr.update(visible=False),
                "Please select an inspection to delete."
            )

        if item["report"] is None:

            return (
                None,
                gr.update(visible=False),
                "Pending inspections cannot be deleted."
            )

        filename = Path(
            item["record"]["source_image"]
        ).name

        message = (
            "**Confirm deletion of this inspection:**\n\n"
            f"`{filename}`\n\n"
            f"Inspection ID: `{inspection_id}`\n\n"
            "The selected record and its associated files "
            "will be archived outside active inspection storage."
        )

        return (
            inspection_id,
            gr.update(visible=True),
            message
        )


def cancel_record_delete():

    return (
        None,
        gr.update(visible=False),
        "Deletion cancelled. No records removed."
    )


def checked_file(value, root):

    path = Path(value)

    if path.is_symlink():
        raise ValueError(
            "Symbolic links are not supported."
        )

    resolved = path.resolve()

    if resolved.parent != root.resolve():

        raise ValueError(
            f"Unexpected file location: {resolved}"
        )

    if not resolved.is_file():

        raise ValueError(
            f"Required file missing: {resolved}"
        )

    return resolved


def file_digest(path):

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):

            digest.update(chunk)

    return digest.hexdigest()


def archive_inspection(inspection_id):

    with HISTORY_LOCK:

        inspections = get_inspections()

        item = inspections.get(inspection_id)

        if item is None:
            raise ValueError(
                "Inspection no longer exists."
            )

        record = item["record"]
        report = item["report"]

        if report is None:
            raise ValueError(
                "Inspection is still being processed."
            )

        record_path = checked_file(
            item["record_path"],
            RECORDS_DIR
        )

        report_path = checked_file(
            item["report_path"],
            ADVISORIES_DIR
        )

        original = checked_file(
            record["source_image"],
            INPUT_DIR
        )

        annotated = checked_file(
            record["annotated_image"],
            ANNOTATED_DIR
        )

        if report.get("source_record_sha256") != file_digest(
            record_path
        ):

            raise ValueError(
                "Inspection report checksum mismatch."
            )

        # Refuse deletion if another record references
        # the same original or annotated image.

        for other_path in RECORDS_DIR.glob(
            "*_inspection.json"
        ):

            if other_path.resolve() == record_path:
                continue

            other_record = read_json(other_path)

            for field, target in [
                ("source_image", original),
                ("annotated_image", annotated)
            ]:

                value = other_record.get(field)

                if value and Path(value).resolve() == target:

                    raise ValueError(
                        "Another inspection references "
                        "the same image. Deletion blocked."
                    )

        DELETED_RECORDS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )

        archive = DELETED_RECORDS_DIR / (
            f"{inspection_id}_{stamp}"
        )

        archive.mkdir(exist_ok=False)

        files = {
            "original": original,
            "annotated": annotated,
            "yolo_record": record_path,
            "llama_report": report_path
        }

        timing_path = record_path.with_suffix(
            ".timing.json"
        )

        if timing_path.exists():

            files["yolo_timing"] = checked_file(
                timing_path,
                RECORDS_DIR
            )

        manifest = {
            "inspection_id": inspection_id,
            "archived_at_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "files": {},
            "database_entry": None
        }

        moved = []

        connection = sqlite3.connect(
            WATCHER_DATABASE.as_uri() + "?mode=rw",
            uri=True,
            timeout=15
        )

        try:

            connection.execute("BEGIN IMMEDIATE")

            row = connection.execute(
                """
                SELECT path, sha256,
                       processed_at, result_folder
                FROM processed
                WHERE path = ?
                """,
                (str(original),)
            ).fetchone()

            if row is None:

                raise ValueError(
                    "Watcher has not finalized this record."
                )

            if row[1] != file_digest(original):

                raise ValueError(
                    "Original image checksum mismatch."
                )

            if Path(row[3]).resolve() != record_path:

                raise ValueError(
                    "Watcher record association mismatch."
                )

            manifest["database_entry"] = {
                "path": row[0],
                "sha256": row[1],
                "processed_at": row[2],
                "result_folder": row[3]
            }

            for label, path in files.items():

                folder = archive / label

                folder.mkdir()

                destination = folder / path.name

                checksum = file_digest(path)

                manifest["files"][label] = {
                    "original_path": str(path),
                    "archive_path": str(destination),
                    "sha256": checksum
                }

                shutil.copy2(path, destination)

                if file_digest(destination) != checksum:

                    raise RuntimeError(
                        "Archive checksum verification failed."
                    )

            manifest_path = archive / "manifest.json"

            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8"
            )

            # All files are archived and verified before
            # removing them from active storage.

            # Remove the input first to prevent reprocessing.
            # Remove the YOLO record before its advisory.

            removal_order = [
                "original",
                "yolo_record",
                "annotated",
                "llama_report"
            ]

            if "yolo_timing" in files:

                removal_order.append("yolo_timing")

            for label in removal_order:

                path = files[label]

                path.unlink()

                moved.append(label)

            cursor = connection.execute(
                "DELETE FROM processed WHERE path = ?",
                (str(original),)
            )

            if cursor.rowcount != 1:

                raise RuntimeError(
                    "Unexpected SQLite deletion count."
                )

            connection.commit()

        except Exception:

            connection.rollback()

            # Restore any removed active files.

            for label in reversed(moved):

                destination = files[label]

                backup = Path(
                    manifest["files"][label]["archive_path"]
                )

                if not destination.exists():

                    shutil.copy2(
                        backup,
                        destination
                    )

            raise

        finally:

            connection.close()

        return archive


def confirm_record_delete(
    pending_id,
    selected_history,
    view_state,
    submission
):

    with HISTORY_LOCK:

        if (
            not pending_id
            or pending_id != selected_history
        ):

            return (
                history_update(selected_history),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "Selection changed. Request deletion again.",
                None,
                gr.update(visible=False),
                gr.skip(),
                *[gr.skip() for _ in range(16)]
            )

        inspections = get_inspections()

        active_id = resolve_active_inspection(
            view_state,
            inspections
        )

        try:

            archive = archive_inspection(
                pending_id
            )

        except Exception as error:

            return (
                history_update(selected_history),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "**Deletion blocked:** "
                + esc(str(error)),
                None,
                gr.update(visible=False),
                gr.skip(),
                *[gr.skip() for _ in range(16)]
            )

        message = (
            "**Selected inspection deleted successfully.** "
            "Recovery archive created: "
            f"`{archive.name}`"
        )

        if active_id == pending_id:

            return (
                history_update(),
                None,
                None,
                None,
                message,
                None,
                gr.update(visible=False),
                "Ready for a new inspection.",
                *blank_view()
            )

        return (
            history_update(),
            view_state,
            submission,
            gr.skip(),
            message,
            None,
            gr.update(visible=False),
            gr.skip(),
            *[gr.skip() for _ in range(16)]
        )



# ============================================================
# GRADIO USER INTERFACE
# ============================================================


def valid_timing(value):

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def inference_fps(record):

    value = record.get("yolo_inference_ms")

    if valid_timing(value) and value > 0:

        return 1000.0 / value

    return None


def performance_ms(value):

    if not valid_timing(value):

        return "Not measured"

    return f"{value:,.3f} ms"


def read_yolo_timing(item):

    if not item:

        return {}

    path = Path(item["record_path"])

    timing_path = path.with_suffix(".timing.json")

    try:

        timing = json.loads(
            timing_path.read_text()
        )

        if timing.get("inspection_id") != (
            item["record"]["inspection_id"]
        ):

            return {}

        if timing.get("source_record_sha256") != (
            file_digest(path)
        ):

            return {}

        return timing

    except (OSError, ValueError, TypeError, KeyError):

        return {}


def total_inspection_seconds(item):

    if not item or item["report"] is None:

        return None

    filename = Path(
        item["record"].get("source_image", "")
    ).name

    match = re.search(
        r"_(\d{8}T\d{12}Z)_[0-9a-f]{10}\.",
        filename
    )

    if match is None:

        return None

    try:

        started = datetime.strptime(
            match.group(1),
            "%Y%m%dT%H%M%S%fZ"
        ).replace(tzinfo=timezone.utc)

        finished = datetime.fromtimestamp(
            Path(
                item["report_path"]
            ).stat().st_mtime,
            tz=timezone.utc
        )

        seconds = (
            finished - started
        ).total_seconds()

        if seconds < 0:

            return None

        return seconds

    except (OSError, ValueError, TypeError):

        return None



def hardware_metric(value, digits=2, unit=""):

    import math

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        return "Not measured"

    return f"{value:.{digits}f}{unit}"


def integrated_performance(item=None):

    report = (
        item.get("report") or {}
        if item else {}
    )

    performance = report.get("performance") or {}

    return (
        performance
        if isinstance(performance, dict)
        else {}
    )


def complete_vision_rate(item=None):

    performance = integrated_performance(item)

    value = performance.get(
        "complete_vision_fps"
    )

    if valid_timing(value) and value > 0:
        return value

    if item:

        timing = read_yolo_timing(item)

        duration = timing.get(
            "yolo_total_processing_ms"
        )

        if valid_timing(duration) and duration > 0:
            return 1000 / duration

    return None


def render_hardware_utilization(item=None):

    if not item:
        return ""

    report = item.get("report")

    if report is None:
        return (
            "### Hardware utilization\n\n"
            "Pending advisory processing."
        )

    hardware = report.get(
        "hardware_utilization"
    ) or {}

    summary = hardware.get("summary")

    if not isinstance(summary, dict):

        return (
            "### Hardware utilization\n\n"
            "**Not measured** for this inspection."
        )

    status = hardware.get("status")

    status_text = {
        "MEASURED": "Measured",
        "PARTIAL_MEASUREMENT":
            "Partial measurement"
    }.get(status, "Measurement unavailable")

    rows = [
        (
            "Average monitored power",
            hardware_metric(
                summary.get(
                    "average_monitored_power_w"
                ), 3, " W"
            )
        ),
        (
            "Peak sampled power",
            hardware_metric(
                summary.get(
                    "peak_sampled_monitored_power_w"
                ), 3, " W"
            )
        ),
        (
            "Estimated sampled energy",
            hardware_metric(
                summary.get(
                    "estimated_monitored_energy_j"
                ), 3, " J"
            )
        ),
        (
            "Estimated sampled energy (Wh)",
            hardware_metric(
                summary.get(
                    "estimated_monitored_energy_wh"
                ), 6, " Wh"
            )
        ),
        (
            "Peak sampled system RAM",
            hardware_metric(
                summary.get(
                    "peak_sampled_system_ram_used_gib"
                ), 3, " GiB"
            )
        ),
        (
            "Minimum available RAM",
            hardware_metric(
                summary.get(
                    "minimum_sampled_available_ram_gib"
                ), 3, " GiB"
            )
        ),
        (
            "Peak sampled CPU temperature",
            hardware_metric(
                summary.get(
                    "peak_sampled_cpu_temperature_c"
                ), 2, " °C"
            )
        ),
        (
            "Sampling coverage",
            hardware_metric(
                hardware.get(
                    "sampling_coverage_percent"
                ), 2, "%"
            )
        ),
        (
            "Valid samples",
            hardware_metric(
                hardware.get("sample_count"), 0
            )
        ),
        (
            "Sampling errors",
            hardware_metric(
                hardware.get(
                    "sampling_error_count"
                ), 0
            )
        ),
        (
            "Requested sampling interval",
            hardware_metric(
                hardware.get(
                    "requested_interval_seconds"
                ), 3, " s"
            )
        ),
        (
            "Sampled duration",
            hardware_metric(
                summary.get(
                    "sampling_duration_seconds"
                ), 3, " s"
            )
        ),
        (
            "Maximum sample gap",
            hardware_metric(
                hardware.get(
                    "maximum_sample_gap_seconds"
                ), 3, " s"
            )
        )
    ]

    peak_ram = summary.get(
        "peak_sampled_system_ram_used_gib"
    )

    available = summary.get(
        "minimum_sampled_available_ram_gib"
    )

    if (
        valid_timing(peak_ram)
        and valid_timing(available)
        and peak_ram + available > 0
    ):

        total_ram = peak_ram + available

        ram_percent = (
            100 * peak_ram / total_ram
        )

        rows.insert(
            5,
            (
                "Peak RAM utilization",
                f"{ram_percent:.2f}%"
            )
        )

    lines = [
        "### Hardware utilization",
        "",
        f"**Measurement status:** {status_text}",
        "",
        "| Hardware metric | Result |",
        "|---|---:|"
    ]

    lines = [
        "### Hardware utilization",
        "",
        f"**Measurement status:** {status_text}",
        "",
        '<div style="display:grid;'
        'grid-template-columns:'
        'repeat(auto-fit,minmax(220px,1fr));'
        'gap:12px;width:100%;margin:12px 0;">'
    ]

    for label, value in rows:

        lines.append(
            '<div style="border:1px solid #94a3b8;'
            'border-radius:10px;padding:14px;'
            'min-width:0;">'
            '<div style="font-size:13px;'
            'opacity:0.8;">'
            + esc(label)
            + '</div>'
            '<div style="font-size:22px;'
            'font-weight:700;margin-top:6px;">'
            + esc(value)
            + '</div></div>'
        )

    lines.extend(["</div>", ""])


    lines.extend([
        "",
        "**Measurement scope:** System-wide monitored PMIC rails.",
        "",
        "Energy is estimated over the sampled interval only.",
        "These values do not represent total USB-C input power "
        "or isolated model consumption.",
        "",
        "Measurement window: vision processing start "
        "to advisory report preparation."
    ])

    return "\n".join(lines)


def render_integrated_processing(item=None):

    if not item:
        return ""

    performance = integrated_performance(item)

    return (
        "### Integrated AI processing\n\n"

        "**Complete vision FPS:** "
        + hardware_metric(
            complete_vision_rate(item), 4
        )
        + "\n\n"

        "**Effective system FPS:** "
        + hardware_metric(
            performance.get(
                "effective_system_fps"
            ), 4
        )
        + "\n\n"

        "**Effective inspections/minute:** "
        + hardware_metric(
            performance.get(
                "effective_inspections_per_minute"
            ), 3
        )
        + "\n\n"

        "**Complete AI processing time:** "
        + hardware_metric(
            performance.get(
                "complete_processing_seconds"
            ), 3, " s"
        )
        + "\n\n"

        "Effective rates are reciprocal latency, "
        "not sustained throughput. AI processing time "
        "excludes upload waiting and final report saving."
    )



def render_performance_header(item=None):

    record = item["record"] if item else {}

    fps = inference_fps(record)

    seconds = total_inspection_seconds(item)

    performance = integrated_performance(item)

    vision_text = hardware_metric(
        complete_vision_rate(item), 3
    )

    system_text = hardware_metric(
        performance.get("effective_system_fps"), 4
    )

    processing_text = hardware_metric(
        performance.get("complete_processing_seconds"),
        2, " s"
    )

    fps_text = (
        f"{fps:.2f}"
        if fps is not None
        else "—"
    )

    if seconds is not None:

        total_text = f"{seconds:.2f} s"

    elif item and item["report"] is None:

        total_text = "Pending"

    else:

        total_text = "—"

    return f"""
    <div id="main-header">

        <div class="header-description">

            <div class="eyebrow">
                ARM BHARAT AI-SOC CHALLENGE
            </div>

            <h1>Intelligent PCB Inspection System</h1>

            <p>
                Automated Visual Defect Detection
                and Engineering Advisory
            </p>

            <p>
                YOLO26n + Fine-tuned Meta Llama 3.2 1B Instruct
            </p>

            <p>
                Raspberry Pi 5 |
                ARM64 CPU |
                Local AI Processing
            </p>

        </div>

        <div class="header-performance"
             style="display:grid;
                    grid-template-columns:repeat(2,minmax(0,1fr));
                    gap:10px;">

            <div class="header-performance-card">

                <div class="header-performance-label">
                    YOLO INFERENCE FPS
                </div>

                <div class="header-fps">
                    {esc(fps_text)}
                </div>

                <div class="header-performance-detail">
                    Equivalent frames/sec
                </div>

            </div>

            <div class="header-performance-card">

                <div class="header-performance-label">
                    COMPLETE VISION FPS
                </div>

                <div class="header-fps">
                    {esc(vision_text)}
                </div>

                <div class="header-performance-detail">
                    Complete YOLO processing
                </div>

            </div>

            <div class="header-performance-card">

                <div class="header-performance-label">
                    EFFECTIVE SYSTEM FPS
                </div>

                <div class="header-fps">
                    {esc(system_text)}
                </div>

                <div class="header-performance-detail">
                    Reciprocal AI processing latency
                </div>

            </div>

            <div class="header-performance-card">

                <div class="header-performance-label">
                    COMPLETE AI PROCESSING TIME
                </div>

                <div class="header-total">
                    {esc(processing_text)}
                </div>

                <div class="header-performance-detail">
                    Vision start to report preparation
                </div>

            </div>

            <div class="header-performance-card"
                 style="grid-column:1 / -1;">

                <div class="header-performance-label">
                    TOTAL INSPECTION TIME
                </div>

                <div class="header-total">
                    {esc(total_text)}
                </div>

                <div class="header-performance-detail">
                    Input copy to saved report (approx.)
                </div>

            </div>

        </div>

    </div>
    """


def render_yolo_record_timing(item=None):

    if not item:

        return ""

    record = item["record"]

    timing = read_yolo_timing(item)

    elapsed = record.get(
        "inference_elapsed_seconds"
    )

    prediction_ms = (
        elapsed * 1000
        if valid_timing(elapsed)
        else None
    )

    fps = inference_fps(record)

    fps_text = (
        f"{fps:.2f} FPS"
        if fps is not None
        else "Not measured"
    )

    return (
        "### YOLO detection record\n\n"

        f"**YOLO inference:** "
        f"{performance_ms(record.get('yolo_inference_ms'))}"
        "\n\n"

        f"**YOLO inference FPS:** {fps_text}"
        "\n\n"

        f"**Total prediction:** "
        f"{performance_ms(prediction_ms)}"
        "\n\n"

        f"**YOLO JSON generation:** "
        f"{performance_ms(record.get('yolo_json_generation_ms'))}"
        "\n\n"

        f"**YOLO JSON saving:** "
        f"{performance_ms(timing.get('yolo_json_save_ms'))}"
        "\n\n"

        f"**Total YOLO processing:** "
        f"{performance_ms(timing.get('yolo_total_processing_ms'))}"
    )


def render_llama_record_timing(item=None):

    if not item:

        return ""

    report = item["report"]

    if report is None:

        return "### Llama advisory record\n\nPending"

    count = report.get(
        "llama_timing_sample_count", 0
    )

    total = (
        report.get(
            "total_measured_llama_inference_ms"
        )
        if count
        else None
    )

    return (
        "### Llama advisory record\n\n"

        f"**Average Llama inference:** "
        f"{performance_ms(report.get('average_llama_inference_ms'))}"
        "\n\n"

        f"**Total Llama inference:** "
        f"{performance_ms(total)}"
        "\n\n"

        f"**Llama advisory pipeline:** "
        f"{performance_ms(report.get('advisory_pipeline_ms'))}"
        "\n\n"

        f"**Measured advisories:** {count}"
    )


# Extend the original loader without rewriting its logic.

_original_load_inspection = load_inspection


def load_inspection(
    inspection_id,
    previous_defect=None,
    inspections=None
):

    if inspections is None:

        inspections = get_inspections()

    item = inspections.get(inspection_id)

    original_outputs = _original_load_inspection(
        inspection_id,
        previous_defect,
        inspections
    )

    return (
        *original_outputs,
        render_performance_header(item),
        (
            render_yolo_record_timing(item)
            + "\n\n"
            + render_integrated_processing(item)
        ),
        (
            render_llama_record_timing(item)
            + "\n\n"
            + render_hardware_utilization(item)
        )
    )



with gr.Blocks(
    title="Intelligent PCB Inspection System",
    analytics_enabled=False
) as demo:

    performance_header = gr.HTML(
        value=render_performance_header()
    )

    service_status = gr.HTML()

    # New PCB image submission panel

    gr.Markdown("## Submit new PCB image")

    with gr.Group():

        pcb_upload = gr.File(
            label="Upload raw PCB image",
            file_count="single",
            file_types=[
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            ],
            type="filepath",
            interactive=True
        )

        submit_button = gr.Button(
            "Submit PCB for Inspection",
            variant="primary"
        )

        submission_status = gr.Markdown(
            value=(
                "Ready for image submission. "
                "Select a raw PCB image to begin."
            )
        )

    submission_state = gr.State(None)

    view_state = gr.State(None)

    pending_delete_state = gr.State(None)

    refresh_button = gr.Button(
        "Refresh Inspection",
        variant="secondary"
    )

    gr.Markdown("## Previous inspection records")

    inspection_selector = gr.Dropdown(
        label="Select previous inspection",
        choices=[],
        value=None,
        interactive=True
    )

    with gr.Row():

        load_record_button = gr.Button(
            "Load Record",
            variant="primary"
        )

        delete_record_button = gr.Button(
            "Delete Record",
            variant="stop"
        )

    history_message = gr.Markdown()

    with gr.Group(visible=False) as delete_confirmation:

        gr.Markdown(
            "### Confirm selected record deletion"
        )

        with gr.Row():

            confirm_delete_button = gr.Button(
                "Confirm Delete",
                variant="stop"
            )

            cancel_delete_button = gr.Button(
                "Cancel",
                variant="secondary"
            )

    metrics = gr.HTML(
        value=render_metrics()
    )

    inspection_notice = gr.Markdown()

    with gr.Row():

        original_image = gr.Image(
            label="Original PCB image",
            interactive=False,
            height=420
        )

        annotated_image = gr.Image(
            label="YOLO annotated PCB image",
            interactive=False,
            height=420
        )

    gr.Markdown("## Detected PCB defects")

    defect_table = gr.Dataframe(
        headers=TABLE_COLUMNS,
        datatype=[
            "str",
            "str",
            "str",
            "number",
            "number",
            "str"
        ],
        value=empty_table(),
        interactive=False,
        wrap=True
    )

    gr.Markdown("## AI-assisted engineering advisory")

    defect_selector = gr.Dropdown(
        label="Select detected defect",
        choices=[],
        interactive=True
    )

    with gr.Row():

        with gr.Column():

            advisory_display = gr.HTML()

        with gr.Column():

            advisory_json = gr.JSON(
                label="Structured six-field advisory JSON"
            )

            advisory_metadata = gr.Markdown()

    with gr.Accordion(
        "Detailed inference performance",
        open=False
    ):

        timing_display = gr.Markdown()

    gr.Markdown("## Inspection report downloads")

    with gr.Row(equal_height=True):

        with gr.Column(scale=1, min_width=280):

            yolo_download = gr.File(
                label="Original YOLO inspection JSON",
                interactive=False
            )

        with gr.Column(scale=1, min_width=280):

            integrated_download = gr.File(
                label="Integrated Llama advisory JSON",
                interactive=False
            )

    gr.Markdown("---")

    gr.Markdown("## AI processing and hardware performance")

    with gr.Column():

        yolo_record_timing = gr.Markdown()

    gr.Markdown("---")

    with gr.Column():

        llama_record_timing = gr.Markdown()

    gr.Markdown("""
    ---

    **Automatic offline PCB inspection system**

    Images are processed by the existing YOLO watcher.

    Advisory generation is handled by the local Llama bridge.

    Llama outputs are checked against the configured
    engineering knowledge base.

    Uploaded images are saved into the existing
    monitored input directory.

    This dashboard submits images and reads saved results.
    It does not repeat AI inference.

    All engineering advisories require further assessment.
    """)

    report_state = gr.State(None)

    signature_state = gr.State(None)

    refresh_timer = gr.Timer(
        value=REFRESH_SECONDS,
        active=True
    )

    inspection_outputs = [
        metrics,
        original_image,
        annotated_image,
        defect_table,
        defect_selector,
        report_state,
        advisory_display,
        advisory_json,
        advisory_metadata,
        yolo_download,
        integrated_download,
        inspection_notice,
        timing_display,
        performance_header,
        yolo_record_timing,
        llama_record_timing
    ]

    refresh_inputs = [
        inspection_selector,
        view_state,
        defect_selector,
        signature_state,
        submission_state
    ]

    refresh_outputs = [
        inspection_selector,
        service_status,
        signature_state,
        submission_status,
        *inspection_outputs
    ]

    # A new browser session starts blank.
    # Only a submitted or explicitly loaded inspection
    # is displayed by automatic refresh.

    demo.load(
        fn=final_refresh,
        inputs=refresh_inputs,
        outputs=refresh_outputs,
        concurrency_id="history_operations",
        concurrency_limit=1
    )

    refresh_timer.tick(
        fn=final_refresh,
        inputs=refresh_inputs,
        outputs=refresh_outputs,
        show_progress="hidden",
        concurrency_id="history_operations",
        concurrency_limit=1
    )

    refresh_button.click(
        fn=reset_dashboard,
        inputs=[],
        outputs=[
            inspection_selector,
            view_state,
            submission_state,
            signature_state,
            pcb_upload,
            submission_status,
            history_message,
            pending_delete_state,
            delete_confirmation,
            *inspection_outputs
        ],
        concurrency_id="history_operations",
        concurrency_limit=1
    )

    submit_button.click(
        fn=submit_new_inspection,
        inputs=[pcb_upload],
        outputs=[
            view_state,
            submission_state,
            signature_state,
            pcb_upload,
            submission_status,
            history_message,
            pending_delete_state,
            delete_confirmation,
            *inspection_outputs
        ],
        concurrency_id="history_operations",
        concurrency_limit=1
    )

    load_record_button.click(
        fn=load_previous_record,
        inputs=[inspection_selector],
        outputs=[
            view_state,
            submission_state,
            signature_state,
            history_message,
            submission_status,
            pending_delete_state,
            delete_confirmation,
            *inspection_outputs
        ],
        concurrency_id="history_operations",
        concurrency_limit=1
    )

    delete_record_button.click(
        fn=request_record_delete,
        inputs=[inspection_selector],
        outputs=[
            pending_delete_state,
            delete_confirmation,
            history_message
        ],
        concurrency_id="history_operations",
        concurrency_limit=1
    )

    cancel_delete_button.click(
        fn=cancel_record_delete,
        inputs=[],
        outputs=[
            pending_delete_state,
            delete_confirmation,
            history_message
        ],
        concurrency_id="history_operations",
        concurrency_limit=1
    )

    confirm_delete_button.click(
        fn=confirm_record_delete,
        inputs=[
            pending_delete_state,
            inspection_selector,
            view_state,
            submission_state
        ],
        outputs=[
            inspection_selector,
            view_state,
            submission_state,
            signature_state,
            history_message,
            pending_delete_state,
            delete_confirmation,
            submission_status,
            *inspection_outputs
        ],
        concurrency_id="history_operations",
        concurrency_limit=1
    )

    defect_selector.input(
        fn=select_advisory,
        inputs=[
            report_state,
            defect_selector
        ],
        outputs=[
            advisory_display,
            advisory_json,
            advisory_metadata
        ]
    )


# ============================================================
# LOCAL OFFLINE DASHBOARD LAUNCH
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("INTELLIGENT PCB INSPECTION DASHBOARD")
    print("=" * 65)

    print("Dashboard URL: http://127.0.0.1:7860")
    print("Automatic refresh:", REFRESH_SECONDS, "seconds")
    print("YOLO records:", RECORDS_DIR)
    print("Llama advisories:", ADVISORIES_DIR)
    print("Cloud sharing: DISABLED")
    print("Analytics: DISABLED")
    print("Duplicate AI inference: DISABLED")
    print("Dashboard image upload: ENABLED")
    print("Upload destination:", INPUT_DIR)

    launch_options = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "share": False,
        "show_error": True,
        "allowed_paths": [
            str(RECORDS_DIR),
            str(ADVISORIES_DIR)
        ]
    }

    launch_parameters = inspect.signature(
        demo.launch
    ).parameters

    if "css" in launch_parameters:
        launch_options["css"] = CSS

    if "theme" in launch_parameters:

        launch_options["theme"] = gr.themes.Soft(
            font=["Arial", "sans-serif"]
        )

    demo.queue(
        default_concurrency_limit=1
    )

    demo.launch(**launch_options)

