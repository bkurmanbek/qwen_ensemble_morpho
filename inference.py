"""
Inference Script for Unified Morphology Model
==============================================

Use the fine-tuned Qwen model for complete morphological analysis.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import argparse
from typing import Dict, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedMorphologyModel:
    """Unified model for complete morphological analysis"""

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
        """Predict complete morphology for a word"""

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
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            logger.warning(f"Raw output: {response[-500:]}")
            return self._get_default_output(word, pos_tag)

    def _extract_json(self, response: str) -> str:
        """Extract JSON from model response"""

        # Try to find JSON in code blocks
        if "```json" in response:
            try:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json_str
            except:
                pass

        if "```" in response:
            try:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json_str
            except:
                pass

        # Try to find JSON by braces
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            return response[start:end]

        return "{}"

    def _get_default_output(self, word: str, pos_tag: str) -> Dict:
        """Return default output when parsing fails"""
        return {
            "POS tag": pos_tag,
            "word": word,
            "lemma": word,
            "morphology": {
                "column": "NaN",
                "дара, негізгі": "NaN",
                "дара, туынды": "NaN",
                "күрделі, біріккен, Бірік.": "NaN",
                "күрделі, қосарланған, Қос.": "NaN",
                "күрделі, қысқарған, Қыс.": "NaN",
                "күрделі, тіркескен, Тірк.": "NaN"
            },
            "semantics": {},
            "lexics": {"мағынасы": "NaN"},
            "sozjasam": {"тәсілін, құрамын шартты қысқартумен беру": "NaN"}
        }


def main():
    """Main inference function"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--word", help="Word to analyze")
    parser.add_argument("--pos_tag", help="POS tag")
    parser.add_argument("--input_file", help="Input JSON file with words to analyze")
    parser.add_argument("--output_file", help="Output JSON file for results")

    args = parser.parse_args()

    # Load model
    model = UnifiedMorphologyModel(args.model_path)
    model.load_model()

    # Single word mode
    if args.word and args.pos_tag:
        logger.info(f"Analyzing: {args.word} ({args.pos_tag})")
        result = model.predict(args.word, args.pos_tag)

        print("\n" + "="*60)
        print("MORPHOLOGICAL ANALYSIS")
        print("="*60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("="*60)

    # Batch mode
    elif args.input_file:
        logger.info(f"Loading input from: {args.input_file}")

        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        results = []

        for i, item in enumerate(input_data):
            word = item.get('word')
            pos_tag = item.get('POS tag')

            if not word or not pos_tag:
                logger.warning(f"Skipping item {i}: missing word or POS tag")
                continue

            logger.info(f"Processing {i+1}/{len(input_data)}: {word}")

            result = model.predict(word, pos_tag)
            results.append(result)

        # Save results
        if args.output_file:
            logger.info(f"Saving results to: {args.output_file}")
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"Processed {len(results)} words")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))

    else:
        print("Please provide either --word and --pos_tag, or --input_file")


if __name__ == "__main__":
    main()
