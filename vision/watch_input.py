#!/usr/bin/env python3
"""Offline, sequential, 0.5-second PCB image watcher; one YOLO model instance."""
import argparse
import hashlib
import json
import logging
import os
import signal
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
STOP = False

def stop(_sig, _frame):
    global STOP
    STOP = True

def region(x, y, w, h):
    col = 'left' if x < w / 3 else ('right' if x >= 2*w/3 else 'center')
    row = 'top' if y < h / 3 else ('bottom' if y >= 2*h/3 else 'center')
    if row == col == 'center': return 'center region'
    if row == 'center': return f'center-{col} region'
    if col == 'center': return f'{row} region'
    return f'{row}-{col} region'

def write_json(path, data):
    temp = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', suffix='.tmp', dir=path.parent, delete=False) as f:
            temp = Path(f.name)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp and temp.exists(): temp.unlink()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def process(model, source, output, weights, conf, imgsz):
    from PIL import Image
    hardware_started = time.monotonic()
    begin = time.perf_counter()
    result_list = model.predict(source=str(source), device='cpu', imgsz=imgsz, conf=conf, save=False, verbose=False)
    elapsed = round(time.perf_counter()-begin, 3)
    if len(result_list) != 1: raise RuntimeError('Expected exactly one image result')
    result = result_list[0]
    height, width = result.orig_shape
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    inspection_id = f'PCB-{stamp}'
    # Two shared output folders, named by source image stem.
    records_dir = output / 'inspection_records'
    images_dir = output / 'annotated_images'
    records_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    suffix = ''
    # Preserve prior inspections if the same image name is processed again.
    while True:
        annotated = images_dir / f'{stem}{suffix}_annotated.png'
        record = records_dir / f'{stem}{suffix}_inspection.json'
        if not annotated.exists() and not record.exists():
            break
        suffix = f'_{inspection_id}' if not suffix else f'_{inspection_id}_{len(suffix)}'
    Image.fromarray(result.plot()[..., ::-1]).save(annotated)
    json_begin = time.perf_counter()
    detections = []
    for i, box in enumerate(result.boxes, 1):
        cls = int(box.cls.item())
        names = result.names
        label = str(names[cls] if isinstance(names, (list, tuple)) else names.get(cls, cls))
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = xyxy
        detections.append({'detection_id': i, 'class_id': cls, 'defect_class': label,
            'confidence': round(float(box.conf.item()), 6),
            'bounding_box_xyxy': [round(v, 2) for v in xyxy],
            'location': region((x1+x2)/2, (y1+y2)/2, width, height),
            'severity': 'NOT_ASSESSED', 'verification_status': 'PENDING_HUMAN_REVIEW'})
    data = {'inspection_id': inspection_id, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'INSPECTION_COMPLETE', 'source_image': str(source), 'annotated_image': str(annotated),
        'model_weights': str(weights), 'device': 'cpu', 'image_width': width, 'image_height': height,
        'confidence_threshold': conf, 'inference_elapsed_seconds': elapsed,
        'yolo_preprocess_ms': result.speed.get('preprocess'),
        'yolo_inference_ms': result.speed.get('inference'),
        'yolo_postprocess_ms': result.speed.get('postprocess'),
        'total_detections': len(detections), 'detections': detections, 'advisory_status': 'NOT_GENERATED'}
    # Measure record construction and serialization.
    json.dumps(data, indent=2, ensure_ascii=False)

    data['yolo_json_generation_ms'] = round(
        (time.perf_counter() - json_begin) * 1000, 4
    )

    save_begin = time.perf_counter()

    write_json(record, data)

    save_ms = (
        time.perf_counter() - save_begin
    ) * 1000

    total_ms = (
        time.perf_counter() - begin
    ) * 1000

    # Keep the published YOLO JSON immutable.
    # Store additional saving measurements separately.

    timing = {
        'inspection_id': inspection_id,
        'source_record_sha256': digest(record),
        'yolo_json_generation_ms':
            data['yolo_json_generation_ms'],
        'yolo_json_save_ms': round(save_ms, 4),
        'yolo_total_processing_ms': round(total_ms, 4),
        'yolo_record_completed_ns': time.time_ns(),
        'vision_started_monotonic': hardware_started,
        'vision_finished_monotonic': time.monotonic()
    }

    timing_path = record.with_suffix('.timing.json')

    try:
        timing['hardware_boot_id'] = Path(
            '/proc/sys/kernel/random/boot_id'
        ).read_text().strip()
        write_json(timing_path, timing)
    except Exception:
        logging.exception(
            'Supplementary timing save failed: %s',
            timing_path
        )

    return record, len(detections)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('/home/berry/pcb_ai/input'))
    ap.add_argument('--weights', type=Path, default=Path('/home/berry/pcb_ai/models/best.pt'))
    ap.add_argument('--output', type=Path, default=Path('/home/berry/pcb_ai/auto_inspections'))
    ap.add_argument('--interval', type=float, default=0.5)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--retry-delay', type=float, default=10.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.interval <= 0 or args.retry_delay <= 0 or not 0 <= args.conf <= 1:
        ap.error('Invalid interval, retry-delay, or confidence')
    source_dir, output, weights = args.input.expanduser().resolve(), args.output.expanduser().resolve(), args.weights.expanduser().resolve()
    if not weights.is_file(): ap.error(f'Missing weights: {weights}')
    source_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / '.watcher_state.sqlite3', timeout=30)
    db.execute('CREATE TABLE IF NOT EXISTS processed (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, processed_at TEXT NOT NULL, result_folder TEXT NOT NULL)')
    db.commit()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    from ultralytics import YOLO
    from PIL import Image
    logging.info('Loading model once: %s', weights)
    model = YOLO(str(weights))
    logging.info('Watching %s every %.1fs; sequential CPU inference', source_dir, args.interval)
    seen = {}  # path -> (size, mtime_ns, stable observations)
    failures = {}  # path -> (signature, retry_after)
    while not STOP:
        try:
            paths = sorted((p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS), key=lambda p: p.name)
            active = set(paths)
            for old in set(seen)-active: seen.pop(old, None); failures.pop(old, None)
            for path in paths:
                if STOP: break
                try:
                    stat = path.stat()
                    signature = (stat.st_size, stat.st_mtime_ns)
                    previous = seen.get(path)
                    stable = previous[2]+1 if previous and previous[:2] == signature else 1
                    seen[path] = (*signature, stable)
                    # Require two matching observations, and at least 0.5 s since last modification.
                    if stable < 2 or time.time_ns()-stat.st_mtime_ns < 500_000_000 or stat.st_size == 0: continue
                    if failures.get(path, (None, 0))[0] == signature and time.monotonic() < failures[path][1]: continue
                    sha = digest(path)
                    row = db.execute('SELECT sha256 FROM processed WHERE path=?', (str(path),)).fetchone()
                    if row and row[0] == sha: continue
                    # Verify image is complete before trying YOLO.
                    with Image.open(path) as img: img.verify()
                    # Recheck after read to avoid processing while copy is in progress.
                    after = path.stat()
                    if (after.st_size, after.st_mtime_ns) != signature: continue
                    logging.info('Inspecting %s', path.name)
                    record, count = process(model, path, output, weights, args.conf, args.imgsz)
                    db.execute('INSERT OR REPLACE INTO processed (path,sha256,processed_at,result_folder) VALUES (?,?,?,?)',
                        (str(path), sha, datetime.now(timezone.utc).isoformat(), str(record)))
                    db.commit()
                    failures.pop(path, None)
                    logging.info('Complete: %s (%d detections) %s', path.name, count, record)
                except (KeyboardInterrupt, SystemExit): raise
                except Exception:
                    logging.exception('Failed to process %s; will retry', path)
                    failures[path] = (seen.get(path, (None, None))[:2], time.monotonic()+args.retry_delay)
        except Exception:
            logging.exception('Watcher scan failed')
        time.sleep(args.interval)
    db.close()
    logging.info('Watcher stopped')

if __name__ == '__main__': main()
