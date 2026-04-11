"""
Download meme images from URLs in the dataset.
Run this on your Mac before uploading to the cluster.

Usage:
    python download_images.py --data /path/to/dataset.json --output ./meme_images
"""

import os
import json
import argparse
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(item, output_dir):
    """Download a single image, save using img_fname."""
    url      = item.get("url")
    fname    = item.get("img_fname")
    post_id  = item.get("post_id", "unknown")

    if not url or not fname:
        return post_id, False, "missing url or img_fname"

    out_path = os.path.join(output_dir, fname)

    # Skip if already downloaded
    if os.path.exists(out_path):
        return post_id, True, "already exists"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        with open(out_path, "wb") as f:
            f.write(response.content)

        return post_id, True, "downloaded"

    except Exception as e:
        return post_id, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to dataset JSON file")
    parser.add_argument("--output", default="./meme_images", help="Output directory for images")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data

    print(f"Total items: {len(items)}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Saving images to: {args.output}")

    # Download in parallel
    success = 0
    failed  = 0
    skipped = 0
    errors  = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_image, item, args.output): item
            for item in items
        }

        for i, future in enumerate(as_completed(futures)):
            post_id, ok, msg = future.result()

            if msg == "already exists":
                skipped += 1
            elif ok:
                success += 1
            else:
                failed += 1
                errors.append(f"{post_id}: {msg}")

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(items)} — "
                      f"Success: {success} | Skipped: {skipped} | Failed: {failed}")

    print(f"\nDone!")
    print(f"  Downloaded : {success}")
    print(f"  Skipped    : {skipped} (already existed)")
    print(f"  Failed     : {failed}")

    if errors:
        print(f"\nFailed downloads:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

        # Save failed list
        with open("failed_downloads.txt", "w") as f:
            f.write("\n".join(errors))
        print(f"\nFull error list saved to failed_downloads.txt")

    # Verify
    downloaded = list(Path(args.output).glob("*"))
    print(f"\nTotal images in output dir: {len(downloaded)}")


if __name__ == "__main__":
    main()
