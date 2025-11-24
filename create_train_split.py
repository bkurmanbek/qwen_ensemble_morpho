#!/usr/bin/env python3
"""
Create training dataset by excluding eval_1000.json items
"""

import json
import sys
from typing import Set, Tuple

def create_unique_key(item: dict) -> Tuple[str, str]:
    """Create unique key from word and POS tag"""
    return (item.get('word', ''), item.get('POS tag', ''))

def create_train_split(
    full_data_path: str,
    eval_data_path: str,
    output_train_path: str
):
    """
    Create training dataset by excluding evaluation items

    Args:
        full_data_path: Path to full dataset
        eval_data_path: Path to evaluation dataset
        output_train_path: Path to save training dataset
    """

    print(f"Loading evaluation data from {eval_data_path}...")
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    print(f"Loaded {len(eval_data)} evaluation items")

    # Create set of unique keys from eval data
    eval_keys: Set[Tuple[str, str]] = set()
    for item in eval_data:
        key = create_unique_key(item)
        eval_keys.add(key)

    print(f"Created {len(eval_keys)} unique eval keys")

    print(f"Loading full data from {full_data_path}...")
    with open(full_data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    print(f"Loaded {len(full_data)} total items")

    # Filter out eval items
    train_data = []
    excluded_count = 0

    for i, item in enumerate(full_data):
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i+1}/{len(full_data)} items...")

        key = create_unique_key(item)
        if key not in eval_keys:
            train_data.append(item)
        else:
            excluded_count += 1

    print(f"\nResults:")
    print(f"  Total items: {len(full_data)}")
    print(f"  Excluded (in eval): {excluded_count}")
    print(f"  Training items: {len(train_data)}")
    print(f"  Evaluation items: {len(eval_data)}")

    # Verify counts
    expected_train = len(full_data) - excluded_count
    assert len(train_data) == expected_train, "Count mismatch!"

    # Save training data
    print(f"\nSaving training data to {output_train_path}...")
    with open(output_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Successfully created training dataset!")
    print(f"  Train: {output_train_path} ({len(train_data):,} items)")
    print(f"  Eval:  {eval_data_path} ({len(eval_data):,} items)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create training dataset by excluding evaluation items"
    )
    parser.add_argument(
        "--full_data",
        default="all_structured_kazakh_data.json",
        help="Path to full dataset"
    )
    parser.add_argument(
        "--eval_data",
        default="eval_1000.json",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output",
        default="train_data.json",
        help="Path to output training dataset"
    )

    args = parser.parse_args()

    create_train_split(
        full_data_path=args.full_data,
        eval_data_path=args.eval_data,
        output_train_path=args.output
    )
