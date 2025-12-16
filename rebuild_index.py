import os
import json
import sys
import time
import httpx
from pathlib import Path

# Configuration
DOCS_DIR = Path("/Users/water/Desktop/docs")
TOTAL_FILES = 927
BASE_URL = "http://localhost:8001"

def extract_text(obj) -> str:
    """Extract text from dict or list of dicts."""
    texts = []
    if isinstance(obj, dict):
        t = obj.get("text") or obj.get("content")
        if t: texts.append(str(t))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content")
                if t: texts.append(str(t))
    return "\n\n".join(texts)

def main():
    if not DOCS_DIR.exists():
        print(f"Error: Directory {DOCS_DIR} does not exist.")
        sys.exit(1)

    print("1. Requesting server to deduplicate existing index...")
    try:
        resp = httpx.post(f"{BASE_URL}/admin/deduplicate", timeout=300)
        resp.raise_for_status()
        print(f"   Deduplication result: {resp.json()}")
    except Exception as e:
        print(f"   Error calling deduplicate: {e}")
        print("   Ensure the server is running with the latest code.")
        return

    print("2. Fetching existing sources...")
    try:
        resp = httpx.get(f"{BASE_URL}/sources", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        existing_sources = set(data.get("sources", []))
        print(f"   Found {len(existing_sources)} existing sources.")
    except Exception as e:
        print(f"   Error fetching sources: {e}")
        return

    print("3. Checking for missing files...")
    missing_files = []
    for i in range(1, TOTAL_FILES + 1):
        filename = f"{i}.json"
        if filename not in existing_sources:
            file_path = DOCS_DIR / filename
            if file_path.exists():
                missing_files.append(file_path)

    print(f"   Found {len(missing_files)} missing files to ingest.")
    
    if not missing_files:
        print("All files are already indexed. Exiting.")
        return

    print("4. Starting batch ingestion...")
    # Process one by one, but wait for server to finish (sync=True)
    # This effectively batches them because we wait for the previous one.
    
    success_count = 0
    fail_count = 0
    
    for idx, file_path in enumerate(missing_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            text = extract_text(data)
            if not text.strip():
                print(f"[{idx}/{len(missing_files)}] Skipping {file_path.name}: Empty text.")
                continue

            payload = {
                "text": text,
                "source": file_path.name,
                "sync": True  # Tell server to wait until done
            }
            
            print(f"[{idx}/{len(missing_files)}] Ingesting {file_path.name}...", end="", flush=True)
            
            # Long timeout because embedding can take time
            resp = httpx.post(f"{BASE_URL}/ingest", json=payload, timeout=120)
            resp.raise_for_status()
            
            print(f" Done. ({resp.json().get('ingested_chunks_count')} chunks)")
            success_count += 1
            
        except Exception as e:
            print(f" Failed: {e}")
            fail_count += 1
            # Optional: sleep a bit on error
            time.sleep(1)

    print(f"\nJob complete. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
