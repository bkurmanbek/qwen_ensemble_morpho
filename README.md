# Kazakh Morphology - Unified Qwen Pipeline

A simplified, single-model approach to complete Kazakh morphological analysis using **Qwen2.5-3B**.

## Overview

This pipeline uses a single fine-tuned Qwen model to perform:
- **Morphology**: Word structure analysis (дара/күрделі, негізгі/туынды)
- **Semantics**: POS-specific semantic categories
- **Lexics**: Word meaning generation (мағынасы)
- **Sozjasam**: Word formation patterns (сөзжасам)

### Advantages

- ✅ **Simpler**: One model instead of three
- ✅ **Faster training**: 4-6 hours total
- ✅ **Easier deployment**: Single model to manage
- ✅ **Lower memory**: 3B parameters vs 7B+14B+3B
- ✅ **Better coherence**: Single model maintains consistency

### Trade-offs

- May have slightly lower accuracy than specialized ensemble
- Less flexibility to update individual components

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Training

#### Full Training (Recommended)

Train on all 1.7M examples:

```bash
bash train.sh
```

#### Quick Test Training

Train on 10,000 samples for testing:

```bash
bash train.sh --max_samples 10000
```

#### Custom Training Options

```bash
# Use 8-bit quantization (saves memory)
bash train.sh --use_8bit

# Custom batch size
bash train.sh --batch_size 2

# Custom number of epochs
bash train.sh --num_epochs 5

# Use a different model size
bash train.sh --model_name Qwen/Qwen2.5-7B-Instruct
```

### 3. Inference

#### Single Word

```bash
python inference.py \
    --model_path ./qwen_unified_model \
    --word "кітап" \
    --pos_tag "Зат есім"
```

#### Batch Processing

Create input JSON file (`test_input.json`):
```json
[
  {"word": "кітап", "POS tag": "Зат есім"},
  {"word": "жылдам", "POS tag": "Сын есім"},
  {"word": "жүгіру", "POS tag": "Етістік"}
]
```

Run batch inference:
```bash
python inference.py \
    --model_path ./qwen_unified_model \
    --input_file test_input.json \
    --output_file results.json
```

### 4. Deployment

Start API server:

```bash
python deployment.py --model_path ./qwen_unified_model --port 8000
```

Test API:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"word": "кітап", "pos_tag": "Зат есім"}'
```

Access API docs: http://localhost:8000/docs

## Project Structure

```
qwen_unified_pipeline/
├── train_unified.py          # Training script
├── inference.py               # Inference script
├── deployment.py              # FastAPI deployment
├── train.sh                   # Training shell script
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── all_structured_kazakh_data.json  (symlink)
└── all_kazakh_grammar_data.json     (symlink)
```

## Training Details

### Model Architecture

- **Base Model**: Qwen/Qwen2.5-3B-Instruct
- **Fine-tuning**: LoRA (r=32, alpha=64)
- **Parameters**: ~20M trainable (0.6% of total)

### Training Configuration

- **Epochs**: 3 (default)
- **Batch Size**: 4 per device (default)
- **Learning Rate**: 2e-4
- **Scheduler**: Cosine
- **Max Length**: 2048 tokens
- **Gradient Accumulation**: 4 steps

### Hardware Requirements

#### Minimum (CPU/MPS)
- 16GB RAM
- Training: ~12 hours
- No GPU required

#### Recommended (GPU)
- RTX 4090 (24GB): 4-6 hours
- A100 (40GB): 3-4 hours
- Can use 8-bit quantization: `--use_8bit`

### Memory Usage

- **Training**: ~12-16GB GPU memory
- **Inference**: ~6-8GB GPU memory
- **CPU Mode**: Works but slower

## Output Format

The model generates complete morphological analysis:

```json
{
  "POS tag": "Зат есім",
  "word": "кітап",
  "lemma": "кітап",
  "morphology": {
    "column": "кітап/Ø",
    "дара, негізгі": "негізгі",
    "дара, туынды": "NaN",
    "күрделі, біріккен, Бірік.": "NaN",
    "күрделі, қосарланған, Қос.": "NaN",
    "күрделі, қысқарған, Қыс.": "NaN",
    "күрделі, тіркескен, Тірк.": "NaN"
  },
  "semantics": {
    "нақты заттар, объектілер": "кітап"
  },
  "lexics": {
    "мағынасы": "кітап -зт. Басылып шыққан еңбек; ақпаратты жинақтаған материалдық зат."
  },
  "sozjasam": {
    "тәсілін, құрамын шартты қысқартумен беру": "зт/Ø"
  }
}
```

## Training Data

- **Total Items**: 1,742,581
- **Valid Training**: 1,719,195 (98.7%)
- **POS Distribution**:
  - Етістік: 1,482,734 (85.1%)
  - Зат есім: 209,343 (12.0%)
  - Сын есім: 39,913 (2.3%)
  - Others: 10,591 (0.6%)

## Performance Tips

### Training
1. Use `--max_samples` for quick testing
2. Use `--use_8bit` to reduce memory usage
3. Reduce `--batch_size` if OOM errors occur
4. Use smaller model (Qwen2.5-1.5B) for faster training

### Inference
1. Batch processing is faster than single predictions
2. Use GPU for better performance
3. Lower temperature (0.1) for consistent output
4. Cache results for repeated queries

## Troubleshooting

### Out of Memory

**During Training**:
```bash
# Reduce batch size
bash train.sh --batch_size 2

# Use 8-bit quantization
bash train.sh --use_8bit

# Use smaller model
bash train.sh --model_name Qwen/Qwen2.5-1.5B-Instruct
```

**During Inference**:
- Close other programs
- Use CPU mode (automatic fallback)
- Process one word at a time

### Slow Training

- Use GPU if available
- Increase batch size if memory allows
- Test with `--max_samples 10000` first

### JSON Parse Errors

The model is trained to output valid JSON. If you get parse errors:
1. Check the raw output
2. Ensure model finished training
3. Try with temperature=0.0 for more deterministic output

## Comparison with Ensemble

| Feature | Unified (This) | Ensemble |
|---------|----------------|----------|
| Models | 1 x 3B | 3 models (7B+14B+3B) |
| Training Time | 4-6 hours | 17-22 hours |
| Memory (Training) | 12-16GB | 24-40GB |
| Memory (Inference) | 6-8GB | 16-24GB |
| Accuracy | Good | Better |
| Deployment | Simple | Complex |
| Cost | Low | High |

## Next Steps

1. Train the model: `bash train.sh`
2. Test inference: `python inference.py --model_path ./qwen_unified_model --word "кітап" --pos_tag "Зат есім"`
3. Deploy to production: `python deployment.py --model_path ./qwen_unified_model`

## Support

- Check training logs for errors
- Ensure all dependencies are installed
- Verify data files exist (symlinks)

## License

Same as parent project.
