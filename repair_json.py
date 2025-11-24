#!/usr/bin/env python3
"""
Repair corrupted JSON files by fixing common issues
Designed to handle large files efficiently
"""

import json
import sys
import os
from pathlib import Path

def repair_json_at_position(file_path, error_pos):
    """Repair JSON file at a specific character position"""

    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"File size: {len(content):,} characters")

    # Get context around error
    start = max(0, error_pos - 1000)
    end = min(len(content), error_pos + 1000)
    context = content[start:end]

    print(f"\nContext around character {error_pos}:")
    print("="*80)
    print(context[error_pos - start - 100:error_pos - start])
    print(">>> ERROR POSITION <<<")
    print(context[error_pos - start:error_pos - start + 100])
    print("="*80)

    # Common fixes
    fixes_applied = []

    # Fix 1: Look for lines without proper indentation
    lines = content.split('\n')
    error_char_count = 0
    error_line_idx = 0

    for i, line in enumerate(lines):
        error_char_count += len(line) + 1  # +1 for newline
        if error_char_count > error_pos:
            error_line_idx = i
            break

    print(f"\nError at line {error_line_idx + 1}")

    # Check nearby lines for indentation issues
    for i in range(max(0, error_line_idx - 2), min(len(lines), error_line_idx + 3)):
        line = lines[i]
        stripped = line.strip()

        # If line starts with a JSON key but has wrong indentation
        if stripped.startswith('"') and ':' in stripped:
            # Check if it should be indented
            if i > 0:
                prev_line = lines[i-1].strip()
                # After a closing brace/bracket or comma, keys should be indented
                if prev_line.endswith(',') or prev_line.endswith('{') or prev_line == '},' :
                    # Count expected indentation (should be 4 or 8 spaces typically)
                    if not line.startswith('    ') and not line.startswith('\t'):
                        print(f"  Line {i+1}: Fixing indentation")
                        lines[i] = '    ' + stripped
                        fixes_applied.append(f"Line {i+1}: Added indentation")

    if fixes_applied:
        print(f"\nApplied {len(fixes_applied)} fixes:")
        for fix in fixes_applied:
            print(f"  - {fix}")

        # Write repaired content
        repaired_content = '\n'.join(lines)

        backup_path = f"{file_path}.backup"
        print(f"\nCreating backup: {backup_path}")
        os.rename(file_path, backup_path)

        print(f"Writing repaired file: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(repaired_content)

        print("✓ File repaired! Testing...")

        # Test if it's valid now
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ JSON is now valid! ({len(data):,} records)")
            return True
        except json.JSONDecodeError as e:
            print(f"✗ Still has errors at line {e.lineno}, column {e.colno}")
            print(f"  Message: {e.msg}")
            print(f"  Position: {e.pos}")
            print("\n  Run the script again to fix the next error")
            return False
    else:
        print("\n✗ No automatic fixes found")
        print("Manual inspection required")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python repair_json.py <json_file> [error_position]")
        print("\nThis script will:")
        print("  1. Create a backup of the original file")
        print("  2. Attempt to fix common JSON errors")
        print("  3. Validate the repaired JSON")
        print("\nIf error_position is not provided, it will try to detect it")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # First, try to detect the error
    print("Validating JSON...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ JSON is already valid! ({len(data):,} records)")
        sys.exit(0)
    except json.JSONDecodeError as e:
        print(f"✗ JSON Error detected:")
        print(f"  Line {e.lineno}, Column {e.colno}")
        print(f"  Message: {e.msg}")
        print(f"  Position: {e.pos}")

        error_pos = e.pos

    # Use provided error position if available
    if len(sys.argv) > 2:
        error_pos = int(sys.argv[2])

    print(f"\nAttempting repair at position {error_pos}...")
    repair_json_at_position(file_path, error_pos)

if __name__ == "__main__":
    main()
