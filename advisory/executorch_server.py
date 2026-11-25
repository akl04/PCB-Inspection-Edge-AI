#!/usr/bin/env python3

import argparse
import json
import logging
import re
import subprocess
import tempfile
import time

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

INFERENCE_LOCK = Lock()

PROJECT = Path("/home/berry/pcb_inspection_final")

RUNNER = Path(
    "/home/berry/executorch/cmake-out-pi-new/"
    "examples/models/llama/llama_main"
)

MODEL = Path(
    "/home/berry/pcb_ai/models/llama/"
    "pcb_llama_int4_ctx2048.pte"
)

TOKENIZER = Path(
    "/home/berry/pcb_ai/models/llama/tokenizer.model"
)

KB = json.loads(
    (PROJECT / "config/pcb_knowledge_base.json")
    .read_text(encoding="utf-8")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def parse_selection(text):

    lines = text.strip().splitlines()

    if not lines:
        return None

    if len(lines) > 2:
        return None

    first = lines[0].strip()

    if first in ("A.", "A)"):
        first = "A"

    elif first in ("B.", "B)"):
        first = "B"

    if first not in ("A", "B"):
        return None

    if len(lines) == 2:
        if lines[1].strip() != "Explanation:":
            return None

    return first



def generate_selection(detection):

    defect = detection["defect"]

    if defect not in KB:
        raise ValueError("Unknown PCB defect class")

    location = str(
        detection.get("location", "Unknown")
    )[:150]

    confidence = float(detection["confidence"])

    if not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")

    actions = KB[defect]["corrective_action"]

    if len(actions) != 3:
        raise ValueError("Expected three candidate actions")

    options = {
        "A": actions[0],
        "B": actions[1]
    }

    prompt = (
        "<|begin_of_text|>"
        "PCB inspection question:\n"
        "The PCB is unpowered.\n"
        f"Defect: {defect}\n"
        "Choose the first inspection action.\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        "Answer:"
    )

    start = time.monotonic()

    with tempfile.TemporaryDirectory(
        prefix="pcb_executorch_"
    ) as work:

        prompt_file = Path(work) / "prompt.txt"

        prompt_file.write_text(
            prompt,
            encoding="utf-8"
        )

        command = [
            str(RUNNER),
            "--model_path=" + str(MODEL),
            "--tokenizer_path=" + str(TOKENIZER),
            "--prompt_file=" + str(prompt_file),
            "--cpu_threads=4",
            "--temperature=0",
            "--seq_len=256",
            "--max_new_tokens=4",
            "--num_bos=0",
            "--num_eos=0",
            "--ignore_eos=false"
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=90,
            cwd=work
        )

    wall_ms = (
        time.monotonic() - start
    ) * 1000

    raw = result.stdout

    observer = {}

    for line in (
        raw + "\n" + result.stderr
    ).splitlines():

        if "PyTorchObserver " in line:

            try:
                observer = json.loads(
                    line.split(
                        "PyTorchObserver ", 1
                    )[1]
                )
            except json.JSONDecodeError:
                pass

    clean = raw.split(
        "PyTorchObserver", 1
    )[0]

    boundary_found = False

    if prompt in clean:

        clean = clean.split(prompt, 1)[1]
        boundary_found = True

    elif prompt.replace(
        "<|begin_of_text|>", "", 1
    ) in clean:

        clean = clean.split(
            prompt.replace(
                "<|begin_of_text|>", "", 1
            ),
            1
        )[1]

        boundary_found = True

    clean = re.split(
        r"<\|(?:eot_id|end_of_text|start_header_id)\|>",
        clean,
        maxsplit=1
    )[0].strip()

    selected = (
        parse_selection(clean)
        if boundary_found else None
    )

    # Only preliminary inspection actions may be
    # prioritized before verification.
    #
    # Option C is a later disposition/rework action
    # in the current knowledge base.

    validated = (
        result.returncode == 0
        and selected in ("A", "B")
    )

    error = None

    if result.returncode != 0:
        error = "ExecuTorch runner failed"

    elif not boundary_found:
        error = "Prompt boundary not found"

    elif selected is None:
        error = "Missing or ambiguous selection"

    def duration(end, begin):

        if end in observer and begin in observer:

            value = (
                observer[end] - observer[begin]
            )

            return value if value >= 0 else None

        return None

    response = {
        "runtime": "ExecuTorch",
        "model": str(MODEL),
        "defect": defect,
        "location": location,
        "confidence": confidence,
        "candidate_actions": options,
        "raw_llama_output": clean,
        "prompt_boundary_found": boundary_found,
        "advisory_method": "LLAMA_ACTION_PRIORITIZATION",
        "selected_option": selected,
        "selected_action": options.get(selected),
        "validated": validated,
        "validation_error": error,
        "return_code": result.returncode,
        "wall_time_ms": round(wall_ms, 2),
        "timings": {
            "prompt_ms": duration(
                "prompt_eval_end_ms",
                "inference_start_ms"
            ),
            "predicted_ms": duration(
                "inference_end_ms",
                "prompt_eval_end_ms"
            ),
            "inference_ms": duration(
                "inference_end_ms",
                "inference_start_ms"
            ),
            "model_load_ms": duration(
                "model_load_end_ms",
                "model_load_start_ms"
            ),
            "ttft_ms": duration(
                "first_token_ms",
                "inference_start_ms"
            ),
            "predicted_n": observer.get(
                "generated_tokens"
            ),
            "predicted_per_second": observer.get(
                "decode_token_per_sec"
            )
        },
        "runtime_metrics": observer
    }

    logging.info(
        "%s | Selection=%s | Validated=%s | %.2f ms",
        defect,
        selected,
        validated,
        wall_ms
    )

    return response


class Handler(BaseHTTPRequestHandler):

    def send_json(self, status, data):

        body = json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):

        if self.path == "/health":

            ready = all(
                path.is_file()
                for path in (
                    RUNNER, MODEL, TOKENIZER
                )
            )

            self.send_json(
                200 if ready else 503,
                {
                    "status": "ok" if ready else "error",
                    "runtime": "ExecuTorch",
                    "model": str(MODEL),
                    "health_scope": "server_and_files"
                }
            )

        else:

            self.send_json(
                404,
                {"error": "Unknown endpoint"}
            )

    def do_POST(self):

        if self.path != "/advisory":

            self.send_json(
                404,
                {"error": "Unknown endpoint"}
            )

            return

        try:

            length = int(
                self.headers.get("Content-Length", "0")
            )

            if not 0 < length <= 16384:

                self.send_json(
                    400,
                    {"error": "Invalid request size"}
                )

                return

            payload = json.loads(
                self.rfile.read(length)
            )

            with INFERENCE_LOCK:
                response = generate_selection(
                    payload["detection"]
                )

            self.send_json(200, response)

        except subprocess.TimeoutExpired:

            self.send_json(
                504,
                {
                    "runtime": "ExecuTorch",
                    "error": "Inference timeout"
                }
            )

        except (ValueError, KeyError, TypeError) as exc:

            self.send_json(
                400,
                {"error": str(exc)}
            )

        except Exception as exc:

            logging.exception("Inference failed")

            self.send_json(
                500,
                {
                    "error": type(exc).__name__,
                    "runtime": "ExecuTorch"
                }
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        type=int,
        default=8091
    )

    args = parser.parse_args()

    for path in (RUNNER, MODEL, TOKENIZER):

        if not path.is_file():
            raise FileNotFoundError(path)

    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        Handler
    )

    logging.info(
        "ExecuTorch advisory server listening on port %d",
        args.port
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
