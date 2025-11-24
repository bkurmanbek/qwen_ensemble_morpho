#!/usr/bin/env python3
"""
Script to find and fix JSON parsing errors
"""

import sys

def find_json_error(file_path, char_pos):
    """Find the context around a JSON error"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Get context around the error
    start = max(0, char_pos - 500)
    end = min(len(content), char_pos + 500)

    context = content[start:end]
    error_pos = char_pos - start

    print(f"File: {file_path}")
    print(f"Character position: {char_pos}")
    print(f"Total file size: {len(content)} characters")
    print("\n" + "="*80)
    print("Context around error (error position marked with >>>):")
    print("="*80)

    # Show context with error marker
    before = context[:error_pos]
    after = context[error_pos:]

    print(before)
    print(">>> ERROR HERE <<<")
    print(after)
    print("="*80)

    # Show lines around the error
    lines = content[:char_pos].split('\n')
    line_num = len(lines)
    col_num = len(lines[-1]) + 1

    print(f"\nLine {line_num}, Column {col_num}")

    # Show 5 lines before and after
    all_lines = content.split('\n')
    start_line = max(0, line_num - 6)
    end_line = min(len(all_lines), line_num + 5)

    print("\nLines around error:")
    for i in range(start_line, end_line):
        marker = ">>> " if i == line_num - 1 else "    "
        print(f"{marker}{i+1:6d}: {all_lines[i]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_json_error.py <json_file> [char_position]")
        sys.exit(1)

    file_path = sys.argv[1]
    char_pos = int(sys.argv[2]) if len(sys.argv) > 2 else 13301813

    find_json_error(file_path, char_pos)
