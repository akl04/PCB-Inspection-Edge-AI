# Raspberry Pi Deployment

## Reference Platform

- Raspberry Pi 5
- Raspberry Pi OS 64-bit
- Arm AArch64 CPU
- Python vision environment
- Python dashboard environment
- Native ExecuTorch Llama runtime

## Deployment Layout

The repository is organized by functionality.

The original application uses the following
deployment directories:

    /home/berry/pcb_ai
    /home/berry/pcb_watcher_05s
    /home/berry/pcb_inspection_final
    /home/berry/executorch

Absolute paths and systemd User settings must
be adapted when using another Linux username.

## Source File Placement

All destination paths below are relative
to /home/berry.

| Repository source | Deployment destination |
|---|---|
| vision/watch_input.py | pcb_watcher_05s/watch_input.py |
| advisory/executorch_server.py | pcb_inspection_final/executorch_server.py |
| integration/llama_bridge.py | pcb_inspection_final/llama_bridge.py |
| integration/wait_for_llama.py | pcb_inspection_final/wait_for_llama.py |
| dashboard/app.py | pcb_inspection_final/dashboard/app.py |
| config/*.json | pcb_inspection_final/config/ |
| hardware_monitoring/*.py | pcb_inspection_final/ |
| deployment/launcher/ | pcb_inspection_final/launcher/ |

## Software Dependencies

The original environment inventories are
available in the requirements directory.

Platform-specific dependency adaptations
may be necessary.

## ExecuTorch Runtime

The original runtime source commit is recorded
in executorch-commit.txt.

The runtime build enables:

- Release optimization
- XNNPACK
- Optimized CPU kernels
- Quantized kernels
- Portable kernels
- KleidiAI support
- Shared workspace support

Expected native Llama runner:

    /home/berry/executorch/cmake-out-pi-new/examples/models/llama/llama_main

The compatible runtime and model artifacts
must be installed before starting the system.

## Background Services

Service definitions are provided under
deployment/services.

The application uses:

- pcb-yolo-watcher
- pcb-llama-server
- pcb-llama-bridge
- pcb-dashboard
- pcb-hardware-monitor

After installing the source, dependencies,
runtime and models, install the corresponding
service definitions in /etc/systemd/system/.

Reload systemd:

    sudo systemctl daemon-reload

Enable and start the services:

    sudo systemctl enable --now pcb-llama-server
    sudo systemctl enable --now pcb-yolo-watcher
    sudo systemctl enable --now pcb-hardware-monitor
    sudo systemctl enable --now pcb-llama-bridge
    sudo systemctl enable --now pcb-dashboard

## Health Verification

Check the ExecuTorch server:

    curl http://127.0.0.1:8091/health

Check the dashboard:

    curl -I http://127.0.0.1:7860/

## Application Usage

1. Open http://127.0.0.1:7860.
2. Upload a PCB image.
3. Wait for defect detection.
4. Wait for advisory processing.
5. Review the annotated PCB image.
6. Review defect-specific recommendations.
7. Review hardware and performance information.
8. Download inspection reports.

## Standalone Application

The launcher opens the dashboard in Chromium
application mode.

The desktop shortcut and application icon
are available in deployment/launcher.

## Deployment Status

The original Raspberry Pi deployment is
operational.

A clean installation from this repository
alone has not yet been independently validated.
