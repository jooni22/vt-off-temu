import os

# --- 1. Clone Repository ---
# Clones the project repository from GitHub. This ensures we have the latest code.
print("Cloning the TEMU-VTOFF repository...")
repo_url = "https://github.com/davidelobba/TEMU-VTOFF.git"
repo_dir = "TEMU-VTOFF"
if not os.path.exists(repo_dir):
    os.system(f"git clone {repo_url}")
else:
    print(f"Directory '{repo_dir}' already exists. Skipping clone.")
os.chdir(repo_dir)
print(f"Changed directory to: {os.getcwd()}")

# --- 2. Install Dependencies with UV ---
# uv is a high-performance Python package installer. It's much faster than pip.
print("\nInstalling uv package manager...")
os.system("pip install uv")

print("\nInstalling project dependencies using uv...")
os.system("uv pip install -r requirements.txt")

# --- 3. Run Optimized Inference ---
# This command executes the optimized inference script with recommended settings
# for performance and quality (e.g., fp16 mixed precision).
print("\nRunning the optimized inference script...")
inference_command = """
python inference.py \\
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \\
    --pretrained_model_name_or_path_sd3_tryoff="stankiem/sd3-tryoff-v0.1" \\
    --example_image="examples/example1.jpg" \\
    --output_dir="output_optimized" \\
    --height=1024 \\
    --width=1024 \\
    --num_inference_steps=28 \\
    --guidance_scale=7.0 \\
    --mixed_precision="fp16" \\
    --seed=42
"""

print(f"Executing command:\n{inference_command}")
os.system(inference_command.strip())

print("\nInference complete. Check the 'output_optimized' directory for results.")
