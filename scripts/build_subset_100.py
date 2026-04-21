"""Deterministic stratified 100-instance subset of SWE-bench Verified.

Strategy:
1. Group by repo.
2. Proportional allocation (largest-remainder method) targeting 100 total.
3. Enforce >= 1 per repo, donating from the largest stratum (django) to zero-quota repos.
4. Deterministic pick: sort each repo's instances by instance_id, shuffle with seed=42, take quota.

Writes:
- data/subset_100.txt      (one instance_id per line)
- data/subset_100.regex    (^(id1|id2|...)$ for use with --filter)
- data/subset_100.json     (metadata: allocation, counts, seed)
"""
from __future__ import annotations
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 42
TARGET = 100
OUT_DIR = Path("data")


def allocate(repo_counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(repo_counts.values())
    # Base allocation via floor(target * share).
    base = {r: math.floor(target * c / total) for r, c in repo_counts.items()}
    remainder = target - sum(base.values())
    # Largest remainder: sort by fractional part desc, tie-break by repo name asc.
    fracs = sorted(
        repo_counts.items(),
        key=lambda rc: (-(target * rc[1] / total - math.floor(target * rc[1] / total)), rc[0]),
    )
    for r, _ in fracs[:remainder]:
        base[r] += 1
    # Guarantee >= 1 per repo. Donate from the largest stratum.
    zeros = [r for r, q in base.items() if q == 0]
    for r in zeros:
        donor = max(base.items(), key=lambda kv: kv[1])[0]
        base[donor] -= 1
        base[r] += 1
    assert sum(base.values()) == target
    return base


def pick_ids(instances_by_repo: dict[str, list[str]], quotas: dict[str, int]) -> list[str]:
    picked: list[str] = []
    for repo in sorted(quotas):
        ids_sorted = sorted(instances_by_repo[repo])
        rng = random.Random(SEED)
        rng.shuffle(ids_sorted)
        picked.extend(ids_sorted[: quotas[repo]])
    return sorted(picked)


def main() -> None:
    ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    by_repo: dict[str, list[str]] = defaultdict(list)
    for row in ds:
        by_repo[row["repo"]].append(row["instance_id"])
    repo_counts = {r: len(ids) for r, ids in by_repo.items()}
    quotas = allocate(repo_counts, TARGET)
    ids = pick_ids(by_repo, quotas)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "subset_100.txt").write_text("\n".join(ids) + "\n")
    regex = "^(" + "|".join(ids) + ")$"
    (OUT_DIR / "subset_100.regex").write_text(regex + "\n")
    (OUT_DIR / "subset_100.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "target": TARGET,
                "dataset": "princeton-nlp/SWE-Bench_Verified",
                "split": "test",
                "repo_counts_full": repo_counts,
                "quotas": quotas,
                "picked_counts": dict(Counter(i.split("__", 1)[0] for i in ids)),
                "instance_ids": ids,
            },
            indent=2,
        )
    )
    print(f"wrote {len(ids)} ids to {OUT_DIR}/subset_100.*")
    for r in sorted(quotas, key=lambda r: -quotas[r]):
        print(f"  {r:30s}  {quotas[r]:3d}  ({100*repo_counts[r]/sum(repo_counts.values()):.1f}% full -> {quotas[r]}% subset)")


if __name__ == "__main__":
    main()
