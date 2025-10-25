import argparse
import torch
import os
import gc
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from src.transformer_vtoff import SD3Transformer2DModel
from src.transformer_sd3_garm import (
    SD3Transformer2DModel as SD3Transformer2DModel_feature_extractor,
)
from src.pipeline_stable_diffusion_3_tryoff_masked import (
    StableDiffusion3TryOffPipelineMasked,
)
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from torchvision import transforms


def load_text_encoders(class_one, class_two, class_three, args):
    text_encoder_one, text_encoder_two, text_encoder_three = None, None, None
    if class_one is not None and class_two is not None:
        text_encoder_one = class_one.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        text_encoder_two = class_two.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=args.revision,
            variant=args.variant,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)

    if class_three is not None:
        text_encoder_three = class_three.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_3",
            revision=args.revision,
            variant=args.variant,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        text_encoder_three.requires_grad_(False)
    return text_encoder_one, text_encoder_two, text_encoder_three


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def main(args):
    """
    Generate virtual try-off image using SD3 pipeline with optimized memory usage.

    Args:
        args: Parsed command line arguments containing model paths, and generation parameters.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # Check GPU and set device
    if torch.cuda.is_available() and "NVIDIA" in torch.cuda.get_device_name(0):
        device = torch.device("cuda:0")
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM Available: {total_vram_gb:.1f} GB")

        if args.max_vram_gb:
            if args.max_vram_gb < total_vram_gb:
                fraction = args.max_vram_gb / total_vram_gb
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                print(
                    f"✓ Attempting to limit VRAM to {args.max_vram_gb:.1f} GB (Fraction: {fraction:.2f})"
                )
            else:
                print(
                    f"⚠ Requested VRAM limit ({args.max_vram_gb} GB) is >= total VRAM ({total_vram_gb:.1f} GB). Ignoring."
                )
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU - this will be very slow!")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print(f"✓ Using precision: {args.mixed_precision} ({weight_dtype})")
    print(f"✓ Resolution: {args.width}x{args.height}")
    print(f"✓ Inference steps: {args.num_inference_steps}")
    print("")

    # Clear cache before loading models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print("Loading tokenizers...")
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
        low_cpu_mem_usage=True,
    )
    print("✓ Tokenizers loaded")

    print("Loading text encoders...")
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
    )
    print("✓ Text encoders loaded")

    print("Loading scheduler...")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    print("✓ Scheduler loaded")

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    print("✓ VAE loaded")

    print("Loading transformers...")
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path_sd3_tryoff,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )

    transformer_vton_feature_extractor = (
        SD3Transformer2DModel_feature_extractor.from_pretrained(
            args.pretrained_model_name_or_path_sd3_tryoff,
            subfolder="transformer_vton",
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True,
        )
    )
    print("✓ Transformers loaded")

    print("Loading image encoders...")
    image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=weight_dtype)

    image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=weight_dtype)
    print("✓ Image encoders loaded")

    print("Building pipeline...")
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

    pipeline.to(device, dtype=weight_dtype)
    print("✓ Pipeline ready")

    # Enable memory-efficient attention
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("✓ Memory-efficient attention (xformers) enabled.")
    except Exception:
        print("⚠ Could not enable xformers, proceeding without it.")

    # Clear cache after loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(
            f"✓ GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )
        print(f"✓ GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    print("")
    print("=" * 60)
    print("Starting inference...")
    print("=" * 60)

    image_path = args.example_image
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_directory = os.path.dirname(image_path)

    print(f"Loading input image: {image_path}")
    image = Image.open(image_path)
    image = image.resize((args.width, args.height))

    caption_path = os.path.join(image_directory, f"{image_name}_caption.txt")
    print(f"Loading caption: {caption_path}")
    caption = open(caption_path, "r").read()
    print(f"Caption: {caption}")

    fine_mask_path = os.path.join(image_directory, f"{image_name}_fine_mask.jpg")
    print(f"Loading fine mask: {fine_mask_path}")
    image_fine_mask = Image.open(fine_mask_path)
    image_fine_mask = image_fine_mask.resize((args.width, args.height))

    binary_mask_path = os.path.join(image_directory, f"{image_name}_binary_mask.jpg")
    print(f"Loading binary mask: {binary_mask_path}")
    image_binary_mask = Image.open(binary_mask_path)
    image_binary_mask = image_binary_mask.resize((args.width, args.height))

    generator = (
        torch.Generator(device=device).manual_seed(args.seed) if args.seed else None
    )
    print(f"✓ Random seed: {args.seed}")
    print("")

    # Manually encode prompt to allow for text encoder offloading
    print("Encoding prompt...")
    # Note: Assuming the pipeline has a .encode_prompt() method similar to diffusers' SD3.
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=caption,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=args.guidance_scale > 1.0,
        negative_prompt="",  # No negative prompt used in original script
    )
    print("✓ Prompt encoded")

    # Offload text encoders to CPU
    print("Offloading text encoders to CPU to save VRAM...")
    if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
        pipeline.text_encoder.to("cpu")
    if hasattr(pipeline, "text_encoder_2") and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.to("cpu")
    if hasattr(pipeline, "text_encoder_3") and pipeline.text_encoder_3 is not None:
        pipeline.text_encoder_3.to("cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(
            f"✓ GPU Memory after offload: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

    print(f"Generating image... (this may take several minutes)")
    with torch.inference_mode():
        image = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            vton_image=image,
            mask_input=image_binary_mask,
            image_input_masked=image_fine_mask,
        ).images[0]

    output_path = f"{args.output_dir}/{image_name}_output.png"
    image.save(output_path)

    print("")
    print("=" * 60)
    print("✓ SUCCESS!")
    print("=" * 60)
    print(f"✓ Output saved to: {output_path}")
    print(f"✓ Image size: {image.size}")

    if torch.cuda.is_available():
        print(
            f"✓ Final GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated"
        )
        print(
            f"✓ Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB peak"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEMU-VTOFF Optimized Inference")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path_sd3_tryoff",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=False,
        help="Output image width (default: 512 for low-end GPU)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        required=False,
        help="Output image height (default: 640 for low-end GPU)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_inference",
        required=False,
        help="Directory to save output images",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        required=False,
        help="Mixed precision type (fp16 recommended for low-end GPU)",
    )
    parser.add_argument(
        "--example_image",
        type=str,
        default="examples/example1.jpg",
        required=False,
        help="Path to input image",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        required=False,
        help="Guidance scale for generation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        required=False,
        help="Number of denoising steps (10 for fast, 28 for quality)",
    )
    parser.add_argument(
        "--max_vram_gb",
        type=float,
        default=None,
        required=False,
        help="Attempt to limit VRAM usage to this amount in GB.",
    )
    args = parser.parse_args()
    main(args)
