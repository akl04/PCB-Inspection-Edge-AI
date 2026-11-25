#!/usr/bin/env python3

import json
import time
import sqlite3
import logging
import argparse
import math

from pathlib import Path
from hardware_monitor import (
    sample_hardware,
    summarize_samples
)
from inspection_sampler import calculate_performance

PROJECT = Path(__file__).resolve().parent
DATABASE = PROJECT / "hardware_benchmarks" / "hardware_samples.sqlite3"

INTERVAL = 0.25
RETENTION_SECONDS = 7200

BOOT_ID = Path(
    "/proc/sys/kernel/random/boot_id"
).read_text().strip()


def connect(database=DATABASE):

    connection = sqlite3.connect(
        database,
        timeout=2
    )

    return connection


def initialize(database=DATABASE):

    database = Path(database)

    database.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with connect(database) as db:

        db.execute("PRAGMA journal_mode=WAL")

        db.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                boot_id TEXT NOT NULL,
                monotonic REAL NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (boot_id, monotonic)
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                boot_id TEXT NOT NULL,
                monotonic REAL NOT NULL,
                message TEXT NOT NULL
            )
        """)


def store_sample(sample, database=DATABASE):

    with connect(database) as db:

        db.execute(
            """
            INSERT OR REPLACE INTO samples
            VALUES (?, ?, ?)
            """,
            (
                BOOT_ID,
                sample["monotonic_seconds"],
                json.dumps(
                    sample,
                    allow_nan=False
                )
            )
        )


def store_error(error, database=DATABASE):

    with connect(database) as db:

        db.execute(
            "INSERT INTO errors VALUES (?, ?, ?)",
            (
                BOOT_ID,
                time.monotonic(),
                str(error)[:500]
            )
        )


def cleanup(database=DATABASE):

    cutoff = time.monotonic() - RETENTION_SECONDS

    with connect(database) as db:

        for table in ("samples", "errors"):

            db.execute(
                f"""
                DELETE FROM {table}
                WHERE boot_id != ?
                OR monotonic < ?
                """,
                (BOOT_ID, cutoff)
            )


def summarize_window(
    started,
    finished,
    boot_id=BOOT_ID,
    database=DATABASE
):

    result = {
        "status": "NOT_MEASURED",
        "measurement_scope": "monitored_pmic_rails",
        "whole_device_input_power": False,
        "system_wide_measurement": True,
        "requested_interval_seconds": INTERVAL,
        "sampling_coverage_percent": 0,
        "sample_count": 0,
        "summary": None,
        "samples": []
    }

    try:

        if boot_id != BOOT_ID:
            raise ValueError("Different system boot")

        if not (
            math.isfinite(started)
            and math.isfinite(finished)
            and finished > started
        ):
            raise ValueError("Invalid observation window")

        duration = finished - started

        result["observation_duration_seconds"] = duration

        with connect(database) as db:

            rows = db.execute(
                """
                SELECT payload FROM samples
                WHERE boot_id = ?
                AND monotonic BETWEEN ? AND ?
                ORDER BY monotonic
                """,
                (boot_id, started, finished)
            ).fetchall()

            errors = db.execute(
                """
                SELECT message FROM errors
                WHERE boot_id = ?
                AND monotonic BETWEEN ? AND ?
                """,
                (boot_id, started, finished)
            ).fetchall()

        samples = [
            json.loads(row[0])
            for row in rows
        ]

        result["samples"] = samples
        result["sample_count"] = len(samples)
        result["sampling_error_count"] = len(errors)

        if len(samples) < 2:
            return result

        summary = summarize_samples(samples)

        coverage = (
            100
            * summary["sampling_duration_seconds"]
            / duration
        )

        gaps = [
            samples[i]["monotonic_seconds"]
            - samples[i-1]["monotonic_seconds"]
            for i in range(1, len(samples))
        ]

        maximum_gap = max(gaps)

        result.update({
            "summary": summary,
            "sampling_coverage_percent":
                min(100, coverage),
            "maximum_sample_gap_seconds":
                maximum_gap,
            "status": (
                "MEASURED"
                if (
                    coverage >= 90
                    and not errors
                    and maximum_gap <= INTERVAL * 3
                )
                else "PARTIAL_MEASUREMENT"
            )
        })

    except Exception as error:

        result["error"] = str(error)

    return result


def run_monitor(seconds=None, database=DATABASE):

    initialize(database)
    cleanup(database)

    started = time.monotonic()
    last_cleanup = started

    while (
        seconds is None
        or time.monotonic() - started < seconds
    ):

        cycle_start = time.monotonic()

        try:

            store_sample(
                sample_hardware(),
                database
            )

        except Exception as error:

            logging.warning(
                "Hardware sampling error: %s",
                error
            )

            try:
                store_error(error, database)
            except Exception:
                logging.exception(
                    "Could not record sampling error"
                )

        if time.monotonic() - last_cleanup >= 60:

            cleanup(database)
            last_cleanup = time.monotonic()

        delay = INTERVAL - (
            time.monotonic() - cycle_start
        )

        if delay > 0:
            time.sleep(delay)

    return summarize_window(
        started,
        time.monotonic(),
        database=database
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seconds",
        type=float,
        default=None
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    result = run_monitor(
        args.seconds,
        args.database
    )

    print(json.dumps({
        key: value
        for key, value in result.items()
        if key != "samples"
    }, indent=2))


if __name__ == "__main__":
    main()
