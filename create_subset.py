import json
import sys

def create_subset(input_path, output_path, num_items=100):
    print(f"Reading from {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            # Read line by line to avoid loading everything
            content = ""
            items_found = 0
            
            # Start of list
            line = f.readline()
            if not line.strip().startswith('['):
                print("Error: File does not start with '['")
                return
            
            content += "[\n"
            
            buffer = ""
            brace_count = 0
            started = False
            
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                    
                buffer += line
                
                # Simple brace counting (assumes well-formatted JSON)
                brace_count += stripped.count('{')
                brace_count -= stripped.count('}')
                
                if '{' in stripped:
                    started = True
                
                if started and brace_count == 0:
                    # End of an item
                    items_found += 1
                    
                    # Remove trailing comma if present in buffer
                    if buffer.strip().endswith(','):
                        # Find the last comma and remove it, but keep the structure
                        # Actually, we can just append it to content.
                        # We'll fix the last item later.
                        pass
                        
                    content += buffer
                    buffer = ""
                    started = False
                    
                    if items_found >= num_items:
                        break
            
            # Remove trailing comma from the last item if it exists
            content = content.rstrip()
            if content.endswith(','):
                content = content[:-1]
            
            content += "\n]"
            
            # Validate
            try:
                data = json.loads(content)
                print(f"Successfully created subset with {len(data)} items.")
                
                with open(output_path, 'w', encoding='utf-8') as out:
                    out.write(content)
                print(f"Saved to {output_path}")
                
            except json.JSONDecodeError as e:
                print(f"Error creating subset: {e}")
                # Fallback: try to just load the top N bytes and fix it?
                # No, let's just try to parse what we have.
                print("Attempting to save anyway for inspection...")
                with open(output_path, 'w', encoding='utf-8') as out:
                    out.write(content)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_subset("all_structured_kazakh_data.json", "test_data.json", 100)
