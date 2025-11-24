#!/usr/bin/env python3
"""
Fix JSON indentation and validation errors
"""

import json
import sys

def fix_json_file(input_file, output_file):
    """
    Try to load and re-save JSON to fix formatting issues
    """
    print(f"Reading {input_file}...")

    # Try to find and fix the issue by reading line by line
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Try to identify the specific error location
    try:
        data = json.loads(content)
        print("✓ JSON is already valid!")

        # Re-save with proper formatting
        print(f"Saving formatted JSON to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ Successfully saved to {output_file}")
        return True

    except json.JSONDecodeError as e:
        print(f"✗ JSON Error at line {e.lineno}, column {e.colno}: {e.msg}")
        print(f"  Character position: {e.pos}")

        # Try to fix common issues
        lines = content.split('\n')

        # Check around the error line
        error_line_idx = e.lineno - 1

        print(f"\nLines around error (line {e.lineno}):")
        start = max(0, error_line_idx - 3)
        end = min(len(lines), error_line_idx + 4)

        for i in range(start, end):
            marker = ">>> " if i == error_line_idx else "    "
            print(f"{marker}{i+1:6d}: {lines[i][:100]}")

        # Check for indentation issues on the error line
        if error_line_idx < len(lines):
            error_line = lines[error_line_idx]
            print(f"\nError line (repr): {repr(error_line[:100])}")

            # Check if it's an indentation issue
            if error_line.strip().startswith('"') and not error_line.startswith('    "'):
                print("\n! Detected indentation issue")
                print(f"  Current: {repr(error_line[:50])}")
                print(f"  Should be: {repr('    ' + error_line.strip()[:46])}")

                # Offer to fix
                response = input("\nAttempt automatic fix? (y/n): ")
                if response.lower() == 'y':
                    # Fix the indentation
                    lines[error_line_idx] = '    ' + error_line.strip()

                    # Write back
                    fixed_content = '\n'.join(lines)

                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)

                    print(f"\n✓ Saved potentially fixed version to {output_file}")
                    print("  Please run the script again to check for more errors")
                    return False

        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_json_indentation.py <input_json> [output_json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"{input_file}.fixed"

    fix_json_file(input_file, output_file)
