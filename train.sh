#!/bin/bash
# Unified Training Pipeline for Kazakh Morphology
# ================================================
# Single Qwen2.5-3B model for complete analysis

set -e  # Exit on error

echo "================================================================================"
echo "Kazakh Morphology - Unified Qwen Training Pipeline"
echo "================================================================================"

# Configuration
DATA_PATH="all_structured_kazakh_data.json"
GRAMMAR_PATH="all_kazakh_grammar_data.json"
OUTPUT_DIR="./qwen_unified_model"
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"

# Parse command line arguments
USE_8BIT=""
MAX_SAMPLES=""
BATCH_SIZE=4
NUM_EPOCHS=3

while [[ $# -gt 0 ]]; do
  case $1 in
    --use_8bit)
      USE_8BIT="--use_8bit"
      shift
      ;;
    --max_samples)
      MAX_SAMPLES="--max_samples $2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE=$2
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS=$2
      shift 2
      ;;
    --model_name)
      MODEL_NAME=$2
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--use_8bit] [--max_samples N] [--batch_size N] [--num_epochs N] [--model_name MODEL]"
      exit 1
      ;;
  esac
done

# Check if data files exist
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$GRAMMAR_PATH" ]; then
    echo "Error: Grammar data not found: $GRAMMAR_PATH"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Training Configuration:"
echo "--------------------------------------------------------------------------------"
echo "  Data: $DATA_PATH"
echo "  Grammar: $GRAMMAR_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  8-bit: $([ -n "$USE_8BIT" ] && echo "Yes" || echo "No")"
echo "  Max Samples: $([ -n "$MAX_SAMPLES" ] && echo "$MAX_SAMPLES" || echo "All")"
echo "--------------------------------------------------------------------------------"
echo ""

echo "Starting training..."
echo "This will take approximately 4-6 hours on RTX 4090..."
echo ""

python train_unified.py \
    --data_path $DATA_PATH \
    --grammar_path $GRAMMAR_PATH \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 2e-4 \
    $USE_8BIT \
    $MAX_SAMPLES

echo ""
echo "================================================================================"
echo "Training Complete!"
echo "================================================================================"
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Test the model:"
echo "     python inference.py --model_path $OUTPUT_DIR --word \"кітап\" --pos_tag \"Зат есім\""
echo ""
echo "  2. Run batch inference:"
echo "     python inference.py --model_path $OUTPUT_DIR --input_file test.json --output_file results.json"
echo ""
echo "  3. Deploy to production:"
echo "     python deployment.py --model_path $OUTPUT_DIR --port 8000"
echo ""
