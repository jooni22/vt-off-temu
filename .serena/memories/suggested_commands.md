# Suggested Commands

## Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Load environment variables
source .env

# Install PyTorch (Maxwell GPU compatible)
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install all dependencies
uv pip install -r requirements.txt
```

## Inference Commands

### Single Image Inference
```bash
source venv/bin/activate
source .env

python inference.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
    --pretrained_model_name_or_path_sd3_tryoff "davidelobba/TEMU-VTOFF" \
    --seed 42 \
    --width 768 \
    --height 1024 \
    --output_dir "put here the output path" \
    --mixed_precision "bf16" \
    --example_image "examples/example1.jpg" \
    --guidance_scale 2.0 \
    --num_inference_steps 28
```

### Dataset Inference
```bash
python inference_dataset.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
    --pretrained_model_name_or_path_sd3_tryoff "davidelobba/TEMU-VTOFF" \
    --dataset_name "dresscode" \
    --dataset_root "path/to/dataset" \
    --output_dir "output/path" \
    --phase "test" \
    --order "paired" \
    --category "all" \
    --batch_size 4 \
    --mixed_precision "bf16" \
    --guidance_scale 2.0 \
    --num_inference_steps 28
```

## Preprocessing (Dataset)

### 1. Caption Generation
```bash
python precompute_utils/captioning_qwen.py \
    --pretrained_model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "dresscode" \
    --dataset_root "path/to/dataset" \
    --filename "qwen_captions_2_5.json" \
    --temperatures 0.2
```

### 2. Text Feature Extraction
```bash
python precompute_utils/precompute_text_features.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
    --dataset_name "dresscode" \
    --phase "test" \
    --text_encoders "T5" "CLIP"
```

### 3. Image Feature Extraction
```bash
python precompute_utils/precompute_image_features.py \
    --dataset "dresscode" \
    --phase "test" \
    --batch_size 4
```

## Utility Commands
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# List installed packages
uv pip list

# Check examples
ls -la examples/
```
