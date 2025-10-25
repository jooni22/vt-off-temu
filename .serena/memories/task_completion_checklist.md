# Task Completion Checklist

## After Code Changes
1. ✅ Verify NVIDIA GPU is being used (not CPU/AMD)
2. ✅ Check PyTorch version == 2.6.0
3. ✅ Test with `examples/example1.jpg` first
4. ✅ Verify output quality in output directory
5. ✅ Check logs for errors/warnings

## Before Committing
- Ensure `.env` is in `.gitignore`
- No hardcoded API keys
- Test inference runs successfully

## Testing Strategy
- Manual inference test with examples
- Check generated image quality
- Verify mask processing works
- Monitor GPU memory usage

## Quality Checks
- Generated images match expected resolution (768x1024)
- Garments properly extracted
- No artifacts/distortions
- Feature alignment works correctly

## Performance
- Use mixed_precision="bf16" for faster inference
- Batch size based on GPU VRAM
- Default: 28 inference steps, guidance_scale 2.0
