#!/usr/bin/env python3
"""
Evaluation Script for Unified Morphology Model
==============================================

Evaluate the fine-tuned model on eval_1000.json and calculate metrics.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import argparse
from typing import Dict, List
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MorphologyEvaluator:
    """Evaluator for morphology model"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the fine-tuned model"""
        logger.info(f"Loading model from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Detect device
        if torch.cuda.is_available():
            device_type = "cuda"
            dtype = torch.float16
            logger.info("Using CUDA")
        elif torch.backends.mps.is_available():
            device_type = "mps"
            dtype = torch.float32
            logger.info("Using Apple MPS")
        else:
            device_type = "cpu"
            dtype = torch.float32
            logger.info("Using CPU")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto" if device_type == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        if device_type == "mps":
            self.model = self.model.to("mps")
        elif device_type == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()
        logger.info("Model loaded successfully")

    def predict(self, word: str, pos_tag: str) -> Dict:
        """Predict morphology for a word"""

        if self.model is None:
            self.load_model()

        # Build prompt
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

        # Format as chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")

        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        elif torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON from response
        json_output = self._extract_json(response)

        # Parse JSON
        try:
            result = json.loads(json_output)
            return result
        except json.JSONDecodeError:
            return None

    def _extract_json(self, response: str) -> str:
        """Extract JSON from model response"""

        # Try to find JSON in code blocks
        if "```json" in response:
            try:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json_str
            except (IndexError, ValueError):
                pass

        if "```" in response:
            try:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json_str
            except (IndexError, ValueError):
                pass

        # Try to find JSON by braces
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            return response[start:end]

        return "{}"

    def evaluate(self, eval_data: List[Dict]) -> Dict:
        """
        Evaluate model on evaluation dataset

        Args:
            eval_data: List of evaluation items

        Returns:
            Dictionary with evaluation metrics
        """

        total = len(eval_data)
        correct_pos = 0
        correct_word = 0
        correct_lemma = 0
        parse_errors = 0

        results = []

        logger.info(f"Evaluating on {total} items...")

        for item in tqdm(eval_data, desc="Evaluating"):
            word = item.get('word')
            pos_tag = item.get('POS tag')

            if not word or not pos_tag:
                continue

            # Get prediction
            prediction = self.predict(word, pos_tag)

            if prediction is None:
                parse_errors += 1
                results.append({
                    'word': word,
                    'pos_tag': pos_tag,
                    'ground_truth': item,
                    'prediction': None,
                    'error': 'JSON parse error'
                })
                continue

            # Check POS tag
            if prediction.get('POS tag') == item.get('POS tag'):
                correct_pos += 1

            # Check word
            if prediction.get('word') == item.get('word'):
                correct_word += 1

            # Check lemma
            if prediction.get('lemma') == item.get('lemma'):
                correct_lemma += 1

            results.append({
                'word': word,
                'pos_tag': pos_tag,
                'ground_truth': item,
                'prediction': prediction,
                'matches': {
                    'pos_tag': prediction.get('POS tag') == item.get('POS tag'),
                    'word': prediction.get('word') == item.get('word'),
                    'lemma': prediction.get('lemma') == item.get('lemma')
                }
            })

        # Calculate metrics
        metrics = {
            'total': total,
            'parse_errors': parse_errors,
            'pos_accuracy': correct_pos / total if total > 0 else 0,
            'word_accuracy': correct_word / total if total > 0 else 0,
            'lemma_accuracy': correct_lemma / total if total > 0 else 0,
            'results': results
        }

        return metrics


def main():
    """Main evaluation function"""

    parser = argparse.ArgumentParser(
        description="Evaluate unified morphology model"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--eval_data",
        default="eval_1000.json",
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit evaluation samples (for testing)"
    )

    args = parser.parse_args()

    # Load evaluation data
    logger.info(f"Loading evaluation data from: {args.eval_data}")
    with open(args.eval_data, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    if args.max_samples:
        eval_data = eval_data[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    # Load model and evaluate
    evaluator = MorphologyEvaluator(args.model_path)
    evaluator.load_model()

    metrics = evaluator.evaluate(eval_data)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total samples:     {metrics['total']}")
    print(f"Parse errors:      {metrics['parse_errors']}")
    print(f"POS accuracy:      {metrics['pos_accuracy']:.2%}")
    print(f"Word accuracy:     {metrics['word_accuracy']:.2%}")
    print(f"Lemma accuracy:    {metrics['lemma_accuracy']:.2%}")
    print("="*80)

    # Save results
    logger.info(f"Saving results to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
