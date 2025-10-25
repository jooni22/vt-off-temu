import os
from huggingface_hub import snapshot_download


def download_all_models():
    """
    Downloads all required models from Hugging Face Hub and saves them to a local directory.
    This script is designed to be run once to set up the local environment.
    """
    # --- Configuration ---
    # A dictionary mapping a descriptive name to the Hugging Face repo ID.
    models_to_download = {
        "sd3-medium-diffusers": "stabilityai/stable-diffusion-3-medium-diffusers",
        "sd3-tryoff-v0.1": "stankiem/sd3-tryoff-v0.1",
        "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
        "clip-vit-bigg-14-laion2b": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    }

    # The target directory to save all models.
    base_models_dir = "models"

    print(
        f"Starting model download process. All models will be saved in the '{base_models_dir}' directory."
    )

    # Create the base directory if it doesn't exist.
    os.makedirs(base_models_dir, exist_ok=True)

    # --- Download Loop ---
    for model_name, repo_id in models_to_download.items():
        local_model_path = os.path.join(base_models_dir, model_name)
        print(f"\n--- Downloading model: {model_name} ---")
        print(f"Source: {repo_id}")
        print(f"Destination: {local_model_path}")

        if os.path.exists(local_model_path):
            print("Model directory already exists. Skipping download.")
            print("If you need to re-download, please delete the directory first.")
            continue

        try:
            # Using snapshot_download to get the entire repository.
            # This is robust and supports resuming interrupted downloads.
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_model_path,
                resume_download=True,
                local_dir_use_symlinks=False,  # Use False for better portability and to avoid symlink issues
            )
            print(f"Successfully downloaded {model_name}.")
        except Exception as e:
            print(f"An error occurred while downloading {model_name}: {e}")
            print(
                "Please check your internet connection and Hugging Face Hub credentials."
            )
            print(
                "You might need to log in using 'huggingface-cli login' in your terminal."
            )

    print("\n-----------------------------------------")
    print("All models have been downloaded.")
    print(
        f"You can now run the inference script pointing to the '{base_models_dir}' directory."
    )
    print("-----------------------------------------")


if __name__ == "__main__":
    download_all_models()
