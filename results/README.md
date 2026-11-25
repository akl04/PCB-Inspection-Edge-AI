# Final PCB Inspection Results

Four user-selected PCB images were freshly inspected
on Raspberry Pi 5 on November 25, 2026.

Images were submitted sequentially through the
production inspection pipeline from the terminal.
The same YOLO26n and ExecuTorch advisory services
used by the dashboard processed these images.

## Benchmark Results

| Test case | Detections | YOLO FPS | Processing time (s) | Hardware status |
|---|---:|---:|---:|---|
| [5799_t_029](5799_t_029/) | 1 | 2.341 | 9.931 | MEASURED |
| [6118_b_010](6118_b_010/) | 4 | 2.696 | 34.655 | MEASURED |
| [6417_b_001](6417_b_001/) | 1 | 2.156 | 9.298 | MEASURED |
| [6445_b_001](6445_b_001/) | 5 | 2.315 | 40.947 | MEASURED |

## Aggregate Results

- Total inspections: 4
- Total detections: 11
- Validated Llama selections: 7
- Knowledge base fallbacks: 4
- Mean YOLO inference FPS: 2.377
- Mean complete processing time: 23.708 seconds
- Hardware measurements completed: 4

## Included Evidence

Each inspection folder contains:

- Original PCB image
- Annotated PCB image
- YOLO detection JSON
- YOLO timing JSON
- Integrated advisory JSON
- Hardware measurements and raw samples
- Processing performance JSON
- Inspection-specific journal log
- SHA256 evidence manifest

The complete machine-readable summary is available
in [benchmark_summary.json](benchmark_summary.json).

## Limitations

- Four user-selected demonstration images.
- Not an unbiased accuracy benchmark.
- Validated selections are not engineering accuracy scores.
- FPS represents reciprocal latency.
- Power covers monitored PMIC rails only.
- Hardware measurements include concurrent workloads.
- Complete processing excludes upload wait and final report save.
- Partial hardware measurements are retained and labelled.
- Network isolation was not verified by this collector.
