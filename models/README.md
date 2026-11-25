# Model Artifacts

## Vision Model

Architecture: YOLO26n

Filename:

    best.pt

Deployment path:

    /home/berry/pcb_ai/models/best.pt

## Advisory Model

Architecture: Llama 3.2 1B Instruct

Runtime: ExecuTorch

Filename:

    pcb_llama_int4_ctx2048.pte

Deployment path:

    /home/berry/pcb_ai/models/llama/pcb_llama_int4_ctx2048.pte

## Tokenizer

Filename:

    tokenizer.model

Deployment path:

    /home/berry/pcb_ai/models/llama/tokenizer.model

## Export Configuration

- Quantization: 8da4w
- Quantization group size: 128
- Export context length: 2048
- Requested backend: XNNPACK
- Export dtype override: FP16

The deployed PTE uses an earlier successfully
exported PCB-specific fine-tuned checkpoint.

The latest fine-tuning experiment is a separate
development artifact.

## Model Availability

Model binaries are not currently included.

Model access and redistribution permissions
are pending verification.

The corresponding trained model artifacts
are required for inference.

## Model Integrity

The deployed model checksums are recorded in:

    checksums.sha256

Verify the model artifacts after installation.

All applicable upstream software, model and
dataset licenses must be respected.
