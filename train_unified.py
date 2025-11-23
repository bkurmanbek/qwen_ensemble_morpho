"""
Unified Training Script for Kazakh Morphology
==============================================

Single Qwen2.5-3B model for complete morphological analysis.
Handles morphology, semantics, lexics, and sozjasam in one pass.
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging
import argparse
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedDatasetBuilder:
    """Build training dataset for unified morphology model"""

    def __init__(self, data_path: str, grammar_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        with open(grammar_path, 'r', encoding='utf-8') as f:
            self.grammar_data = json.load(f)

    def build_dataset(self, max_samples: int = None) -> Dataset:
        """Build HuggingFace dataset for complete morphology"""

        examples = []
        skipped_count = 0

        data_to_process = self.data[:max_samples] if max_samples else self.data
        logger.info(f"Processing {len(data_to_process)} items...")

        for i, item in enumerate(data_to_process):
            if (i + 1) % 10000 == 0:
                logger.info(f"  Processed {i+1}/{len(data_to_process)} items...")

            try:
                # Basic validation
                if not isinstance(item, dict):
                    skipped_count += 1
                    continue

                # Check required fields
                if 'word' not in item or 'POS tag' not in item:
                    skipped_count += 1
                    continue

                # Validate all sections
                if not self._validate_item(item):
                    skipped_count += 1
                    continue

                example = self._create_training_example(item)
                if example:
                    examples.append(example)
                else:
                    skipped_count += 1

            except Exception as e:
                skipped_count += 1
                continue

        logger.info(f"Created {len(examples)} training examples")
        logger.info(f"Skipped {skipped_count} invalid/empty items")

        if not examples:
            raise ValueError("No valid training examples found!")

        return Dataset.from_list(examples)

    def _validate_item(self, item: dict) -> bool:
        """Validate that item has sufficient data"""

        # Check morphology
        morph = item.get('morphology', {})
        if not isinstance(morph, dict) or not morph:
            return False

        # At least one morphology field should be set
        has_morph = any(v != 'NaN' and v for v in morph.values())
        if not has_morph:
            return False

        return True

    def _create_training_example(self, item: dict) -> dict:
        """Create a single training example for complete morphology"""

        pos_tag = item['POS tag']
        word = item['word']

        # Get POS-specific semantic fields
        pos_grammar = self.grammar_data.get(pos_tag, {})

        # Build system prompt
        system_prompt = f"""Сіз қазақ тілінің морфологиясы, семантикасы, лексикологиясы және сөзжасамы бойынша сарапшысыз.

МІНДЕТ: Берілген сөздің ТОЛЫҚ МОРФОЛОГИЯЛЫҚ ТАЛДАУЫН жасаңыз.

НҰСҚАУЛАР:
1. ТЕК валидті JSON форматында жауап беріңіз
2. Барлық бөлімдерді толық толтырыңыз:
   - morphology: құрылымдық талдау (column, дара/күрделі, негізгі/туынды)
   - semantics: POS табына сәйкес семантикалық категориялар
   - lexics: сөздің толық мағынасы
   - sozjasam: сөзжасам тәсілін қысқартумен беру
3. lemma анықтаңыз
4. Қазақ тілінде жазыңыз
5. Егер мәлімет жоқ болса, "NaN" қойыңыз

JSON ҚҰРЫЛЫМЫ міндетті түрде сақталуы керек."""

        user_prompt = f"""СӨЗ: {word}
СӨЗ ТАБЫ: {pos_tag}

Толық морфологиялық талдау:"""

        # Build expected output
        output = self._build_output_json(item)

        return {
            'system': system_prompt,
            'user': user_prompt,
            'assistant': output,
            'pos_tag': pos_tag,
            'word': word
        }

    def _build_output_json(self, item: dict) -> str:
        """Build the expected JSON output"""

        output = {
            "POS tag": item['POS tag'],
            "word": item['word'],
            "lemma": item.get('lemma', item['word']),
            "morphology": item.get('morphology', {}),
            "semantics": item.get('semantics', {}),
            "lexics": item.get('lexics', {}),
            "sozjasam": item.get('sozjasam', {})
        }

        return json.dumps(output, ensure_ascii=False, indent=2)


def train_unified_model(
    data_path: str,
    grammar_path: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    use_8bit: bool = False,
    max_samples: int = None
):
    """Train the unified morphology model"""

    # Detect device
    if torch.cuda.is_available():
        device_type = "cuda"
        logger.info("Using CUDA")
    elif torch.backends.mps.is_available():
        device_type = "mps"
        logger.info("Using Apple MPS")
        use_8bit = False  # bitsandbytes doesn't support MPS
    else:
        device_type = "cpu"
        logger.info("Using CPU")
        use_8bit = False

    logger.info("Building dataset...")
    builder = UnifiedDatasetBuilder(data_path, grammar_path)
    dataset = builder.build_dataset(max_samples=max_samples)

    # Split train/val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Format dataset
    def format_fn(example):
        messages = [
            {"role": "system", "content": example['system']},
            {"role": "user", "content": example['user']},
            {"role": "assistant", "content": example['assistant']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    train_dataset = train_dataset.map(format_fn)
    val_dataset = val_dataset.map(format_fn)

    # Load model
    logger.info(f"Loading model: {model_name}")

    if use_8bit and device_type == "cuda":
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        model = prepare_model_for_kbit_training(model)
    elif device_type == "mps":
        # Apple Silicon - load in float32 for stability
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device_type == "cuda" else torch.float32,
            device_map="auto" if device_type == "cuda" else None,
            trust_remote_code=True
        )

    # Configure LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=(device_type == "cuda"),
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if (use_8bit and device_type == "cuda") else "adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,
        max_grad_norm=0.3,
        use_mps_device=(device_type == "mps")
    )

    # Tokenize
    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length"
        )
        result["labels"] = result["input_ids"].copy()
        return result

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to training data JSON")
    parser.add_argument("--grammar_path", required=True, help="Path to grammar data JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory for model")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct", help="Base model to fine-tune")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit training samples (for testing)")

    args = parser.parse_args()

    train_unified_model(
        data_path=args.data_path,
        grammar_path=args.grammar_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_8bit=args.use_8bit,
        max_samples=args.max_samples
    )
