# YOLO26n PCB Defect Detection Training

## Training Platform

The YOLO26n model was trained using a Kaggle GPU
notebook and subsequently deployed on Raspberry Pi 5
for offline CPU inference.

## Included Files

- `yolov26n_training.ipynb`: Original training,
  validation, prediction and model analysis source.
  Historical execution outputs have been removed.
- `pcb_dataset.yaml`: Final nine-class dataset
  configuration from notebook Cell 18.

## Dataset Structure

    Data_YOLO/
        images/
            train/
            val/
        labels/
            train/
            val/

The notebook also examines COCO annotations to
identify dataset category mappings.

Dataset access and redistribution permissions
must be verified separately.

## Final Class Mapping

| ID | Label | Defect |
|---|---|---|
| 0 | SH | Short Circuit |
| 1 | SP | Spur |
| 2 | SC | Spurious Copper |
| 3 | OP | Open Circuit |
| 4 | MB | Mouse Bite |
| 5 | HB | Hole Breakout |
| 6 | CS | Conductor Scratch |
| 7 | CFO | Conductor Foreign Object |
| 8 | BMFO | Base Material Foreign Object |

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | YOLO26n |
| Configured epochs | 50 |
| Image size | 640 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Early stopping patience | 10 |
| Device | GPU 0 |
| Workers | 2 |
| Dataset cache | Disabled |

## Notebook Organization

- Cells 1-17: Environment setup, dataset exploration
  and preliminary class mapping checks.
- Cell 18: Final dataset configuration.
- Cells 19-21: GPU verification and model setup.
- Cell 22: Model training.
- Cell 23: Validation.
- Cells 24-26: Sample image predictions.
- Cells 27-29: Validation plots and training analysis.
- Cells 30-31: Checkpoint and model information.

The preliminary class names in Cells 8 and 9 are
superseded by the final mapping in Cell 18.

## Training Output

Expected best checkpoint:

    /kaggle/working/Results/PCB_YOLO26n/weights/best.pt

## Raspberry Pi Deployment

The deployed vision checkpoint is located at:

    /home/berry/pcb_ai/models/best.pt

The deployed checkpoint checksum is recorded in
`models/checksums.sha256`.

## Reproducibility Notes

The notebook is preserved as supplied, except for
removed execution outputs and notebook metadata.

Dataset paths must be adapted when using a different
training environment.

The original notebook installs Ultralytics without
pinning an exact training package version.

Exact checkpoint reproduction and a checksum match
against the original Kaggle training artifact have
not been independently verified.

The configured epoch count is a maximum training
setting, not proof of the number of completed epochs.

Historical training outputs are excluded from this
repository copy.
