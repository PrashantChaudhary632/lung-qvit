"""
Downloads LUNA16 Subset 0 + required CSVs from Kaggle (avc0706/luna16 mirror).
Reusable on any machine with Kaggle API credentials configured.
"""

import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "avc0706/luna16"
OUT_DIR = Path("data/luna16_subset0")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES_TO_GET = ["annotations.csv", "candidates_V2/candidates_V2.csv"]
SUBSET_PREFIX = "subset0/"


def list_files_with_retry(api, dataset, page_token=None, page_size=200, max_retries=5):
    for attempt in range(max_retries):
        try:
            return api.dataset_list_files(dataset, page_token=page_token, page_size=page_size)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise


api = KaggleApi()
api.authenticate()

print("Listing dataset files (larger page size + retry on rate limits)...")
all_files = []
page_token = None
page_num = 1
while True:
    print(f"  Fetching page {page_num}...")
    result = list_files_with_retry(api, DATASET, page_token=page_token)
    all_files.extend(result.files)
    page_token = getattr(result, "next_page_token", None)
    page_num += 1
    if not page_token:
        break
    time.sleep(0.5)

subset0_files = [f.name for f in all_files if f.name.startswith(SUBSET_PREFIX)]
print(f"Found {len(all_files)} total files, {len(subset0_files)} under {SUBSET_PREFIX}")

to_download = FILES_TO_GET + subset0_files
print(f"Downloading {len(to_download)} files total to {OUT_DIR}...")

for i, fname in enumerate(to_download, 1):
    print(f"[{i}/{len(to_download)}] {fname}")
    for attempt in range(5):
        try:
            api.dataset_download_file(DATASET, file_name=fname, path=str(OUT_DIR), force=False)
            break
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"  FAILED after retries: {fname} — {e}")
                break

print("Done.")