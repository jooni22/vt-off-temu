import argparse
import os
import time
import torch
from PIL import Image
from torchvision import transforms
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    T5EncoderModel,
    PretrainedConfig,
    CLIPTokenizer,
    T5TokenizerFast,
)
from src.pipeline_stable_diffusion_3_tryoff_masked import (
    StableDiffusion3TryOffPipelineMasked,
)
from src.transformer_vtoff import SD3Transformer2DModel
from src.transformer_sd3_garm import (
    SD3Transformer2DModel as SD3Transformer2DModel_feature_extractor,
)
from diffusers.models.modeling_utils import _get_model_file


def load_text_encoders(
    text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
):
    """Loads and returns the three text encoders for the pipeline."""
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_three = text_encoder_cls_three.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_3",
        revision=args.revision,
        variant=args.variant,
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: str,
    subfolder: str = "text_encoder",
):
    """Imports the model class from the pretrained model name or path."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def main(args):
    """
    Generate virtual try-on images using an optimized Stable Diffusion 3 based pipeline.
    This script includes optimizations for performance and memory efficiency.
    """
    # --- 1. Setup Environment and Device ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Adhere to device priority: NVIDIA GPU > CPU.
    if torch.cuda.is_available() and "NVIDIA" in torch.cuda.get_device_name(0):
        device = torch.device("cuda:0")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Warning: NVIDIA GPU not found. Falling back to CPU.")

    # Setup mixed precision
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print(f"Using data type: {weight_dtype}")

    # --- 2. Load Models ---
    print("Loading models...")
    model_load_start_time = time.time()

    # Use torch.no_grad() to reduce memory consumption during model loading
    with torch.no_grad():
        # Load Tokenizers
        # These are distinct tokenizers for each text encoder, not redundant loads.
        tokenizer_one = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
        )
        tokenizer_three = T5TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_3",
            revision=args.revision,
        )

        # Load Text Encoders
        # Stable Diffusion 3 uses three different text encoders (CLIP-L, OpenCLIP-bigG, T5-XXL).
        # The following loads these three distinct models.
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path,
            args.revision,
            subfolder="text_encoder_2",
        )
        text_encoder_cls_three = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path,
            args.revision,
            subfolder="text_encoder_3",
        )

        text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
            text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
        )

        # Load VAE, Scheduler, and Transformer models
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path_sd3_tryoff,
            subfolder="transformer",
            revision=args.revision,
            variant=args.variant,
        )
        transformer_vton_feature_extractor = (
            SD3Transformer2DModel_feature_extractor.from_pretrained(
                args.pretrained_model_name_or_path_sd3_tryoff,
                subfolder="transformer_vton",
                revision=args.revision,
                variant=args.variant,
            )
        )

        # Load Image Encoders
        image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        )

        # --- 3. Assemble and Prepare Pipeline ---
        pipeline = StableDiffusion3TryOffPipelineMasked(
            scheduler=noise_scheduler,
            vae=vae,
            transformer_vton_feature_extractor=transformer_vton_feature_extractor,
            transformer_garm=transformer,
            image_encoder_large=image_encoder_large,
            image_encoder_bigG=image_encoder_bigG,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            tokenizer_3=tokenizer_three,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            text_encoder_3=text_encoder_three,
        )

        print("Moving models to device...")
        pipeline.to(device, dtype=weight_dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Model loading and setup time: {time.time() - model_load_start_time:.2f}s")

    # --- 4. Load and Preprocess Data ---
    print("Loading and preprocessing data...")
    data_load_start_time = time.time()

    image_name = os.path.splitext(os.path.basename(args.example_image))[0]
    image_directory = os.path.dirname(args.example_image)

    # Optimized image loading and transformation
    # Define a single transformation pipeline for resizing
    preprocess_rgb = transforms.Compose(
        [
            transforms.Resize(
                (args.height, args.width),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    preprocess_l = transforms.Compose(
        [
            transforms.Resize(
                (args.height, args.width),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )

    # Load images with error handling
    try:
        image = Image.open(args.example_image).convert("RGB")
        fine_mask_path = os.path.join(image_directory, f"{image_name}_fine_mask.jpg")
        image_fine_mask = Image.open(fine_mask_path).convert("RGB")
        binary_mask_path = os.path.join(
            image_directory, f"{image_name}_binary_mask.jpg"
        )
        image_binary_mask = Image.open(binary_mask_path).convert("L")
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}. Please check the paths.")
        return
    except Exception as e:
        print(f"An error occurred while loading images: {e}")
        return

    # Apply transformations
    image_tensor = preprocess_rgb(image).unsqueeze(0)
    fine_mask_tensor = preprocess_rgb(image_fine_mask)
    binary_mask_tensor = preprocess_l(image_binary_mask)

    # Read caption
    try:
        caption_path = os.path.join(image_directory, f"{image_name}_caption.txt")
        with open(caption_path, "r") as f:
            caption = f.read().strip()
    except FileNotFoundError:
        print(
            f"Warning: Caption file not found at {caption_path}. Using an empty caption."
        )
        caption = ""

    generator = (
        torch.Generator(device=device).manual_seed(args.seed) if args.seed else None
    )

    print(f"Data loading time: {time.time() - data_load_start_time:.2f}s")

    # --- 5. Run Inference ---
    print("Starting inference...")
    inference_start_time = time.time()

    # Use inference_mode for efficiency and autocast for mixed-precision performance
    with (
        torch.inference_mode(),
        torch.cuda.amp.autocast(
            enabled=(device.type == "cuda" and weight_dtype == torch.float16)
        ),
    ):
        output_image = pipeline(
            prompt=caption,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            vton_image=image_tensor,
            mask_input=binary_mask_tensor,
            image_input_masked=fine_mask_tensor,
        ).images[0]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Inference time: {time.time() - inference_start_time:.2f}s")

    # --- 6. Save Output ---
    try:
        output_filename = os.path.join(
            args.output_dir, f"{image_name}_output_{int(time.time())}.png"
        )
        output_image.save(output_filename)
        print(f"Output image saved to: {output_filename}")
    except Exception as e:
        print(f"Failed to save the output image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized inference script for Stable Diffusion 3 Virtual Try-On.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model and Path Arguments
    model_group = parser.add_argument_group("Model and Paths")
    model_group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Base SD3 model path from Hugging Face or local directory.",
    )
    model_group.add_argument(
        "--pretrained_model_name_or_path_sd3_tryoff",
        type=str,
        default="stankiem/sd3-tryoff-v0.1",
        help="Path to the specialized SD3 Try-Off transformer model.",
    )
    model_group.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Specific model version to use (e.g., a git commit hash).",
    )
    model_group.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model variant to load (e.g., 'fp16' for half-precision weights).",
    )

    # Input and Output Arguments
    io_group = parser.add_argument_group("Input and Output")
    io_group.add_argument(
        "--example_image",
        type=str,
        required=True,
        help="Path to the primary input image of the person.",
    )
    io_group.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory where the generated images will be saved.",
    )

    # Generation Parameter Arguments
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--height", type=int, default=1024, help="The height of the output image."
    )
    gen_group.add_argument(
        "--width", type=int, default=1024, help="The width of the output image."
    )
    gen_group.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of denoising steps.",
    )
    gen_group.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Classifier-Free Guidance scale.",
    )
    gen_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible generation.",
    )

    # Performance Arguments
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Enable mixed precision ('fp16' or 'bf16') for memory and speed gains.",
    )

    args = parser.parse_args()
    main(args)
