import os

# Clone the repository
!git clone https://github.com/davidelobba/TEMU-VTOFF.git
os.chdir('TEMU-VTOFF')

# Install dependencies
!pip install -r requirements.txt

# Run inference
!python inference.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --pretrained_model_name_or_path_sd3_tryoff="davidelobba/TEMU-VTOFF" \
    --example_image="examples/example1.jpg" \
    --output_dir="output_inference" \
    --width=768 \
    --height=1024 \
    --guidance_scale=2.0 \
    --num_inference_steps=28
