# JSON Repair Guide

## Problem

If you encounter this error during training:

```
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 455499 column 5
```

This means the `all_structured_kazakh_data.json` file was corrupted during transfer to your training machine.

## Solution Options

### Option 1: Re-transfer the File (Recommended)

The JSON file is valid on the local machine. Simply re-upload it to your training environment:

```bash
# If using Google Colab, re-upload the file
# If using rsync/scp, transfer again with verification:
rsync -avz --checksum all_structured_kazakh_data.json user@remote:/path/to/dest/

# Or use scp with compression:
scp -C all_structured_kazakh_data.json user@remote:/path/to/dest/
```

### Option 2: Repair on Training Machine

If re-transferring is not feasible, use the repair script on your training machine:

```bash
# Copy repair_json.py to your training machine
# Then run:
python repair_json.py all_structured_kazakh_data.json

# This will:
# 1. Create a backup (all_structured_kazakh_data.json.backup)
# 2. Attempt to fix the error
# 3. Validate the repaired JSON
# 4. Tell you if more errors exist (run again if needed)
```

### Option 3: Use the Grammar Data Only (Quick Test)

For a quick test with a smaller file:

```bash
# Modify train.sh to use grammar data for testing
python train_unified.py \
  --data_path all_kazakh_grammar_data.json \
  --grammar_path all_kazakh_grammar_data.json \
  --output_dir ./test_model \
  --max_samples 1000  # Limit for quick test
```

## Verification

After fixing, verify the JSON is valid:

```bash
python -c "import json; data = json.load(open('all_structured_kazakh_data.json')); print(f'✓ Valid! {len(data):,} records')"
```

## File Info

- **Local file**: Valid JSON with 1,742,581 records
- **File size**: ~1.9GB
- **Expected transfer time**: 5-10 minutes on good connection

## Prevention

To prevent corruption during future transfers:

1. Use `rsync` with `--checksum` flag
2. Verify file integrity: `md5sum all_structured_kazakh_data.json`
3. Compare checksums before and after transfer
4. Use compression during transfer to speed up and reduce corruption risk
