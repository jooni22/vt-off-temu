# TEMU-VTOFF Project Overview

## Purpose
- **Virtual Try-Off**: Generate clean in-shop garment images from clothed individuals
- Dual-DiT (Diffusion Transformer) architecture
- Multi-category support: upper-body, lower-body, full-body garments
- Text-enhanced generation with DINOv2 feature alignment

## Tech Stack
- **Python**: Primary language
- **PyTorch 2.6.0**: Deep learning framework (CUDA 12.6)
- **Transformers**: HuggingFace models
- **Diffusers**: SD3 Medium base model
- **CLIP/T5**: Text encoders
- **Qwen2.5-VL**: Captioning model
- **Accelerate**: Multi-GPU training
- **WandB**: Experiment tracking

## Key Dependencies
- torch==2.6.0, torchvision==0.21.0
- diffusers (from git repo)
- transformers, accelerate
- opencv-python-headless
- clean-fid, pytorch-fid, DISTS-pytorch
- deepspeed (optional optimization)

## Project Structure
```
├── src/                    # Core model implementations
│   ├── transformer_vtoff.py           # Main VTOFF transformer
│   ├── transformer_sd3_garm.py        # Feature extractor
│   ├── attention_vton_mixed.py        # Hybrid attention
│   ├── attention_processor_*.py       # Attention processors
│   └── pipeline_*.py                  # SD3 pipelines
├── precompute_utils/       # Dataset preprocessing
├── examples/               # Example images + masks
├── inference.py            # Single image inference
├── inference_dataset.py    # Batch dataset inference
├── metrics.py              # Evaluation metrics
└── requirements.txt
```

## Model Access
- Base: `stabilityai/stable-diffusion-3-medium-diffusers` (HF access required)
- Checkpoint: `davidelobba/TEMU-VTOFF`
