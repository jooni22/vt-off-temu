# Code Style & Conventions

## General Style
- Python 3.x codebase
- Standard Python naming: `snake_case` for functions/variables
- Class names: `PascalCase` (e.g., `SD3Transformer2DModel`)
- Private methods: prefix with `_`

## Type Hints
- Used for function signatures
- Example: `def load_text_encoders(class_one, class_two, class_three):`

## Import Organization
1. Standard library (argparse, os)
2. Third-party (torch, transformers, diffusers)
3. Local modules (src.*)

## Model Architecture Patterns
- Transformer-based models extend base classes
- Pipeline pattern for inference
- Attention processors as separate modules
- Feature extractors separate from generators

## Configuration
- Environment variables via `.env` file
- Command-line args via `argparse`
- Model configs from HuggingFace

## GPU/Device Handling
- **MANDATORY**: Always check for NVIDIA GPU first
- Device priority: `cuda:0` > `cpu`
- Mixed precision: `bf16` or `fp16` support
- Never use AMD GPU for computation

## File Organization
- Source models: `src/`
- Utils: `precompute_utils/`, `utils/`
- Scripts: root level (inference.py, metrics.py)
- Assets: `examples/`, `assets/`
