#!/bin/bash

# Minimal inference script for TEMU-VTOFF with progress monitoring
# Optimized for lower-end GPU (Tesla M40)

echo "=== TEMU-VTOFF Minimal Inference Script ==="
echo ""

# Set HuggingFace cache directory
export HF_HOME="./hf_models"
echo "✓ HF_HOME set to: $HF_HOME"

# Check GPU
echo ""
echo "=== GPU Information ==="
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')"

# Create output directory
mkdir -p output_inference
echo ""
echo "✓ Output directory: output_inference"

# Minimal settings for low-end hardware
WIDTH=512
HEIGHT=640
STEPS=10
GUIDANCE=2.0
PRECISION="fp16"

echo ""
echo "=== Inference Settings (Minimal) ==="
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Inference Steps: ${STEPS}"
echo "Guidance Scale: ${GUIDANCE}"
echo "Precision: ${PRECISION}"
echo ""

echo "=== Starting Inference ==="
echo "Note: First run will download models (~5-10GB). This may take 10-30 minutes."
echo "Models will be cached in ./hf_models for future use."
echo ""

python inference.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
    --pretrained_model_name_or_path_sd3_tryoff "davidelobba/TEMU-VTOFF" \
    --seed 42 \
    --width $WIDTH \
    --height $HEIGHT \
    --output_dir "output_inference" \
    --mixed_precision "$PRECISION" \
    --example_image "examples/example1.jpg" \
    --guidance_scale $GUIDANCE \
    --num_inference_steps $STEPS

if [ $? -eq 0 ]; then
    echo ""
    echo "=== SUCCESS ==="
    echo "✓ Image generated successfully!"
    echo "✓ Output saved to: output_inference/example1_output.png"
    ls -lh output_inference/
else
    echo ""
    echo "=== ERROR ==="
    echo "✗ Inference failed. Check error messages above."
    exit 1
fi
