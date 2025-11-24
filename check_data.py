import json
import sys

def check_json(file_path):
    print(f"Checking {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"SUCCESS: {file_path} is valid JSON. Item count: {len(data)}")
        return True
    except json.JSONDecodeError as e:
        print(f"ERROR: {file_path} is NOT valid JSON.")
        print(f"Error details: {e}")
        return False
    except Exception as e:
        print(f"ERROR: An error occurred while reading {file_path}.")
        print(f"Error details: {e}")
        return False

if __name__ == "__main__":
    files_to_check = [
        "all_structured_kazakh_data.json",
        "all_kazakh_grammar_data.json"
    ]
    
    all_valid = True
    for file_path in files_to_check:
        if not check_json(file_path):
            all_valid = False
            
    if all_valid:
        print("\nAll JSON files are valid.")
        sys.exit(0)
    else:
        print("\nSome JSON files are invalid.")
        sys.exit(1)
