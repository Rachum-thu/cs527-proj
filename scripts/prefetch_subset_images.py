"""Pre-pull SWE-bench docker images for a subset of instance IDs.

Idempotent: images already present in the local docker registry are skipped.
Concurrency is intentionally low (default 4) to avoid network contention that
caused the 27 TimeoutExpired failures we saw with 32 concurrent cold pulls.

Usage:
    python scripts/prefetch_subset_images.py data/subset_100.txt
    python scripts/prefetch_subset_images.py data/subset_100.txt --workers 6
"""
from __future__ import annotations
import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path


IMAGE_TEMPLATE = "docker.io/swebench/sweb.eval.x86_64.{id_safe}:latest"
PULL_TIMEOUT = 1800  # 30 min per image; matplotlib & co. can be 3-5 GB


def image_name_for(instance_id: str) -> str:
    return IMAGE_TEMPLATE.format(id_safe=instance_id.replace("__", "_1776_")).lower()


def is_cached(image: str) -> bool:
    r = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    return r.returncode == 0


def pull(image: str, retries: int = 2) -> tuple[str, str, float]:
    """Pull an image. Returns (image, status, elapsed_seconds). status in {cached, pulled, failed}."""
    if is_cached(image):
        return image, "cached", 0.0
    last_err = ""
    for attempt in range(retries + 1):
        t0 = time.monotonic()
        try:
            r = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                timeout=PULL_TIMEOUT,
            )
            elapsed = time.monotonic() - t0
            if r.returncode == 0:
                return image, "pulled", elapsed
            last_err = (r.stderr or r.stdout or "").strip().splitlines()[-1][:200]
        except subprocess.TimeoutExpired:
            last_err = f"timed out after {PULL_TIMEOUT}s"
        if attempt < retries:
            time.sleep(5)
    return image, f"failed: {last_err}", time.monotonic() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("subset_file", type=Path, help="File with one instance_id per line")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ids = [line.strip() for line in args.subset_file.read_text().splitlines() if line.strip()]
    images = [image_name_for(i) for i in ids]
    print(f"target: {len(images)} images, workers={args.workers}, pull_timeout={PULL_TIMEOUT}s")

    cached = [im for im in images if is_cached(im)]
    todo = [im for im in images if im not in cached]
    print(f"already cached: {len(cached)}/{len(images)}; need to pull: {len(todo)}")

    if not todo:
        print("nothing to do.")
        return 0

    t0 = time.monotonic()
    results: dict[str, tuple[str, float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(pull, im): im for im in todo}
        for fut in concurrent.futures.as_completed(futures):
            im, status, secs = fut.result()
            results[im] = (status, secs)
            n_done = len(results)
            print(f"[{n_done:3d}/{len(todo)}] {status:>8s}  {secs:6.1f}s  {im}")

    elapsed = time.monotonic() - t0
    pulled = [im for im, (s, _) in results.items() if s == "pulled"]
    failed = [(im, s) for im, (s, _) in results.items() if s.startswith("failed")]
    print()
    print(f"done in {elapsed:.1f}s. pulled={len(pulled)}  failed={len(failed)}")
    for im, s in failed:
        print(f"  FAILED: {im} -- {s}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
