import json
import sys

try:
    import ijson
    print("ijson is installed.")
except ImportError:
    print("ijson is NOT installed.")
    sys.exit(0)

def test_ijson(file_path):
    print(f"Testing ijson on {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            items = ijson.items(f, 'item')
            count = 0
            for item in items:
                count += 1
                if count % 10000 == 0:
                    print(f"  Read {count} items...", end='\r')
            print(f"\nSuccess! Read {count} items.")
    except Exception as e:
        print(f"\nError reading {file_path} with ijson: {e}")

if __name__ == "__main__":
    test_ijson("test_data.json")
    # test_ijson("all_structured_kazakh_data.json") # Uncomment to test large file
