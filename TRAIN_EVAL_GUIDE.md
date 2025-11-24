# Training and Evaluation Guide

Complete guide for training and evaluating the Kazakh Morphology model with proper train/eval split.

## Dataset Split

The data is split into two sets:
- **Training set**: `train_data.json` (1,720,552 items)
- **Evaluation set**: `eval_1000.json` (1,004 items)

The evaluation set has been completely excluded from the training data to prevent data leakage.

## Workflow

### Step 1: Create Train/Eval Split (Already Done!)

The split has already been created for you. If you need to recreate it:

```bash
python create_train_split.py \
    --full_data all_structured_kazakh_data.json \
    --eval_data eval_1000.json \
    --output train_data.json
```

This will:
- Load `eval_1000.json` (evaluation set)
- Load `all_structured_kazakh_data.json` (full dataset)
- Remove all items in eval set from full dataset
- Save the result to `train_data.json`

**Note**: The script uses (word, POS tag) as unique identifier, so duplicates are automatically handled.

### Step 2: Train the Model

Train on `train_data.json` (excluding eval set):

```bash
# Option 1: Using the training script (recommended)
bash train.sh

# Option 2: Direct Python call
python train_unified.py \
    --data_path train_data.json \
    --grammar_path all_kazakh_grammar_data.json \
    --output_dir ./qwen_unified_model \
    --num_epochs 3 \
    --batch_size 4
```

**Training options:**
- `--use_8bit`: Enable 8-bit quantization (CUDA only)
- `--max_samples N`: Train on first N samples (for testing)
- `--batch_size N`: Batch size per device (default: 4)
- `--num_epochs N`: Number of training epochs (default: 3)
- `--model_name MODEL`: Base model to use (default: Qwen/Qwen2.5-4B-Instruct)

### Step 3: Evaluate on eval_1000.json

After training, evaluate on the held-out evaluation set:

```bash
python evaluate.py \
    --model_path ./qwen_unified_model \
    --eval_data eval_1000.json \
    --output evaluation_results.json
```

**Evaluation options:**
- `--max_samples N`: Evaluate on first N samples (for quick testing)

This will:
- Load your trained model
- Run inference on all items in `eval_1000.json`
- Calculate accuracy metrics (POS tag, word, lemma)
- Save detailed results to `evaluation_results.json`

**Output metrics:**
- **Total samples**: Number of evaluation items
- **Parse errors**: JSON parsing failures
- **POS accuracy**: Percentage of correct POS tags
- **Word accuracy**: Percentage of correct words
- **Lemma accuracy**: Percentage of correct lemmas

### Step 4: Analyze Results

View the evaluation results:

```bash
# Quick summary
cat evaluation_results.json | jq '.total, .pos_accuracy, .lemma_accuracy'

# Detailed results
cat evaluation_results.json | jq '.results[] | select(.matches.lemma == false)'
```

## File Structure

```
qwen_unified_pipeline/
├── all_structured_kazakh_data.json    # Full dataset (1.7M items)
├── eval_1000.json                     # Evaluation set (1K items)
├── train_data.json                    # Training set (1.7M items, excluding eval)
├── all_kazakh_grammar_data.json       # Grammar definitions
├── create_train_split.py              # Script to create train/eval split
├── train_unified.py                   # Training script
├── train.sh                           # Training wrapper
├── evaluate.py                        # Evaluation script
├── inference.py                       # Inference on single words
└── deployment.py                      # Production API server
```

## Important Notes

### Data Integrity
- ✓ Evaluation set is **completely excluded** from training
- ✓ No data leakage between train and eval
- ✓ Unique identification by (word, POS tag) tuple

### Training on Corrupted Data
If your `all_structured_kazakh_data.json` has corruption issues:

1. **Option A**: Install `ijson` for streaming (recommended)
   ```bash
   pip install ijson
   ```
   The training script will automatically skip corrupted records.

2. **Option B**: Repair the JSON file
   ```bash
   python repair_json.py all_structured_kazakh_data.json
   ```

### Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (with 8-bit quantization)
- RAM: 32GB
- Storage: 10GB free space

**Recommended:**
- GPU: RTX 4090 (24GB VRAM)
- RAM: 64GB
- Storage: 50GB free space

**Training time:**
- Full dataset (1.7M): ~4-6 hours on RTX 4090
- Small test (10K): ~15-20 minutes on RTX 4090

## Quick Testing

Test the full pipeline with a small subset:

```bash
# 1. Train on 1000 samples
python train_unified.py \
    --data_path train_data.json \
    --grammar_path all_kazakh_grammar_data.json \
    --output_dir ./test_model \
    --max_samples 1000 \
    --num_epochs 1 \
    --batch_size 2

# 2. Evaluate on 10 samples
python evaluate.py \
    --model_path ./test_model \
    --eval_data eval_1000.json \
    --output test_results.json \
    --max_samples 10

# 3. View results
cat test_results.json | jq
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 2 or 1)
- Use `--use_8bit` flag (CUDA only)
- Use `--max_samples` to train on subset

### Connection Errors
- Check internet connection for model download
- Use cached models if available
- Set HuggingFace cache: `export HF_HOME=/path/to/cache`

### Parse Errors in Evaluation
- Model outputs invalid JSON
- Increase `temperature` in evaluate.py
- Try more training epochs
- Check training data quality

## Next Steps

After successful evaluation:

1. **Deploy to production:**
   ```bash
   python deployment.py --model_path ./qwen_unified_model --port 8000
   ```

2. **Run inference on new data:**
   ```bash
   python inference.py \
       --model_path ./qwen_unified_model \
       --input_file new_data.json \
       --output_file predictions.json
   ```

3. **Fine-tune further:**
   - Adjust learning rate
   - Try different epochs
   - Experiment with batch size
