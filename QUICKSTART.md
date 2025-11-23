# Quick Start - Unified Qwen Pipeline

Get started with the unified Qwen morphology model in 5 minutes.

## Setup (1 minute)

```bash
cd qwen_unified_pipeline

# Install dependencies
pip install -r requirements.txt
```

## Option 1: Quick Test (1-2 hours)

Train on a small sample to test the pipeline:

```bash
bash train.sh --max_samples 10000 --num_epochs 1
```

This creates a model trained on 10K samples in ~1-2 hours.

## Option 2: Full Training (4-6 hours)

Train on all 1.7M samples:

```bash
bash train.sh
```

## Test the Model

```bash
# Single word prediction
python inference.py \
    --model_path ./qwen_unified_model \
    --word "кітап" \
    --pos_tag "Зат есім"
```

## Deploy to Production

```bash
# Start API server
python deployment.py \
    --model_path ./qwen_unified_model \
    --port 8000
```

Test the API:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"word": "кітап", "pos_tag": "Зат есім"}'
```

## Key Differences from Ensemble

| Feature | Unified | Ensemble |
|---------|---------|----------|
| **Training Time** | 4-6 hours | 17-22 hours |
| **Memory Usage** | 12-16GB | 24-40GB |
| **Models to Manage** | 1 | 3 |
| **Deployment** | Simple | Complex |
| **Accuracy** | Good | Better |

## Common Commands

```bash
# Train with 8-bit quantization (saves memory)
bash train.sh --use_8bit

# Train with custom batch size
bash train.sh --batch_size 2

# Use different model size
bash train.sh --model_name Qwen/Qwen2.5-7B-Instruct

# Process multiple words
python inference.py \
    --model_path ./qwen_unified_model \
    --input_file test.json \
    --output_file results.json
```

## Troubleshooting

### Out of Memory?
```bash
# Use smaller batch size
bash train.sh --batch_size 2

# Or use 8-bit quantization
bash train.sh --use_8bit
```

### Slow Training?
- Use GPU if available
- Test with `--max_samples 1000` first

## What This Pipeline Does

The unified model performs **complete morphological analysis** in one pass:

1. **Morphology**: Word structure (дара/күрделі, негізгі/туынды)
2. **Semantics**: POS-specific categories
3. **Lexics**: Word meaning (мағынасы)
4. **Sozjasam**: Word formation patterns

All in a **single 3B parameter model** using **LoRA fine-tuning**.

## Next Steps

- Full documentation: See [README.md](README.md)
- API documentation: http://localhost:8000/docs (after deployment)
- Compare with ensemble: See parent folder

## Support

All Python files have been validated for syntax. If you encounter issues:

1. Check requirements are installed
2. Verify data files exist (symlinks)
3. Check available memory (12GB+ recommended)

That's it! You're ready to train and deploy. 🚀
