import argparse
import json
import math
import re
import subprocess
import time

from datetime import datetime, timezone
from pathlib import Path


EXPECTED_RAILS = (
    "3V7_WL_SW",
    "3V3_SYS",
    "1V8_SYS",
    "DDR_VDD2",
    "DDR_VDDQ",
    "1V1_SYS",
    "0V8_SW",
    "VDD_CORE",
    "3V3_DAC",
    "3V3_ADC",
    "0V8_AON",
    "HDMI",
)


def run_command(*args):
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )

    return result.stdout.strip()


def read_pmic():
    output = run_command(
        "vcgencmd", "pmic_read_adc"
    )

    currents = {}
    voltages = {}

    pattern = re.compile(
        r"^\s*(\S+)\s+"
        r"(current|volt)\(\d+\)="
        r"([-+]?\d+(?:\.\d+)?)([AV])\s*$"
    )

    for line in output.splitlines():
        match = pattern.match(line)

        if not match:
            continue

        name, kind, value, unit = match.groups()

        value = float(value)

        if not math.isfinite(value):
            continue

        if kind == "current" and unit == "A":
            currents[name.removesuffix("_A")] = value

        elif kind == "volt" and unit == "V":
            voltages[name.removesuffix("_V")] = value

    rails = {}

    for name in EXPECTED_RAILS:

        if name not in currents or name not in voltages:
            continue

        voltage = voltages[name]
        current = currents[name]

        if voltage < 0 or current < 0:
            continue

        rails[name] = {
            "voltage_v": voltage,
            "current_a": current,
            "power_w": voltage * current,
        }

    missing = [
        name for name in EXPECTED_RAILS
        if name not in rails
    ]

    if missing:
        raise RuntimeError(
            "Missing or invalid PMIC rails: "
            + ", ".join(missing)
        )

    total_power = sum(
        rail["power_w"]
        for rail in rails.values()
    )

    return {
        "measurement_scope": "monitored_pmic_rails",
        "whole_device_input_power": False,
        "rail_count": len(rails),
        "rails": rails,
        "monitored_power_w": total_power,
        "core_rail_power_w": rails["VDD_CORE"]["power_w"],
        "external_5v_voltage_v": voltages.get("EXT5V"),
    }


def read_memory():
    values = {}

    with open("/proc/meminfo") as file:

        for line in file:

            key, value = line.split(":", 1)

            values[key] = int(value.strip().split()[0])

    total = values["MemTotal"]

    available = values["MemAvailable"]

    used = total - available

    return {
        "total_ram_gib": total / 1048576,
        "available_ram_gib": available / 1048576,
        "used_ram_gib": used / 1048576,
        "ram_utilization_percent": 100 * used / total,
        "swap_used_gib": (
            values["SwapTotal"] - values["SwapFree"]
        ) / 1048576,
    }


def read_temperature():
    path = Path(
        "/sys/class/thermal/thermal_zone0/temp"
    )

    return float(path.read_text().strip()) / 1000


def sample_hardware():

    started = time.monotonic()

    power = read_pmic()

    finished = time.monotonic()

    return {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "monotonic_seconds": (
            started + finished
        ) / 2,

        "power": power,

        "memory": read_memory(),

        "cpu_temperature_c": read_temperature(),
    }


def summarize_samples(samples):

    if len(samples) < 2:
        raise RuntimeError(
            "At least two valid samples are required"
        )

    powers = [
        sample["power"]["monitored_power_w"]
        for sample in samples
    ]

    times = [
        sample["monotonic_seconds"]
        for sample in samples
    ]

    duration = times[-1] - times[0]

    if duration <= 0:
        raise RuntimeError("Invalid sampling duration")

    energy = 0.0

    for index in range(1, len(samples)):

        interval = times[index] - times[index - 1]

        energy += (
            powers[index - 1] + powers[index]
        ) / 2 * interval

    return {
        "measurement_scope": "monitored_pmic_rails",

        "whole_device_input_power": False,

        "sample_count": len(samples),

        "sampling_duration_seconds": duration,

        "average_monitored_power_w": energy / duration,

        "peak_sampled_monitored_power_w": max(powers),

        "minimum_sampled_monitored_power_w": min(powers),

        "estimated_monitored_energy_j": energy,

        "estimated_monitored_energy_wh": energy / 3600,

        "peak_sampled_system_ram_used_gib": max(
            sample["memory"]["used_ram_gib"]
            for sample in samples
        ),

        "minimum_sampled_available_ram_gib": min(
            sample["memory"]["available_ram_gib"]
            for sample in samples
        ),

        "peak_sampled_cpu_temperature_c": max(
            sample["cpu_temperature_c"]
            for sample in samples
        ),

        "average_sampled_core_rail_power_w": sum(
            sample["power"]["core_rail_power_w"]
            for sample in samples
        ) / len(samples),
    }


def benchmark(seconds, interval):

    if seconds <= 0 or interval <= 0:
        raise ValueError(
            "Duration and interval must be positive"
        )

    samples = []

    start = time.monotonic()

    deadline = start + seconds

    next_sample = start

    print("Sampling hardware...", flush=True)

    while True:

        sample = sample_hardware()

        samples.append(sample)

        elapsed = time.monotonic() - start

        print(
            f"Sample {len(samples):03d} | "
            f"{elapsed:6.2f}s | "
            f"Power: "
            f"{sample['power']['monitored_power_w']:.3f} W | "
            f"RAM: "
            f"{sample['memory']['used_ram_gib']:.3f} GiB | "
            f"Temperature: "
            f"{sample['cpu_temperature_c']:.1f} C",
            flush=True,
        )

        if time.monotonic() >= deadline:
            break

        next_sample += interval

        delay = min(next_sample, deadline) - time.monotonic()

        if delay > 0:
            time.sleep(delay)

    return {
        "summary": summarize_samples(samples),
        "samples": samples,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seconds",
        type=float,
        default=15,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    result = benchmark(
        args.seconds,
        args.interval,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = args.output.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(result, indent=2)
    )

    temporary.replace(args.output)

    print("\nHARDWARE BENCHMARK SUMMARY")

    print(
        json.dumps(
            result["summary"],
            indent=2,
        )
    )

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

