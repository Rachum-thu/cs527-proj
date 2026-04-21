"""Aggregate all experimental data for the paper.

Produces a comprehensive JSON with per-instance and summary statistics
for both Text and Optical conditions on 100 SWE-bench Verified instances.
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_TRAJ_DIR = PROJECT_ROOT / "results" / "text"
OPTICAL_TRAJ_DIR = PROJECT_ROOT / "results" / "optical"
SUBSET_FILE = PROJECT_ROOT / "data" / "subset_100.txt"
OUTPUT = PROJECT_ROOT / "results" / "paper_data.json"


def extract_text_instance(iid: str) -> dict:
    traj_file = TEXT_TRAJ_DIR / iid / f"{iid}.traj.json"
    if not traj_file.exists():
        return {"instance_id": iid, "condition": "text", "error": "no trajectory"}

    traj = json.loads(traj_file.read_text())
    info = traj["info"]
    msgs = traj.get("messages", [])

    # Token counts from message usage fields
    total_input = 0
    total_output = 0
    for m in msgs:
        usage = m.get("extra", {}).get("response", {}).get("usage", {})
        total_input += usage.get("prompt_tokens", 0)
        total_output += usage.get("completion_tokens", 0)

    # Wall clock from timestamps
    timestamps = [m.get("extra", {}).get("timestamp", 0) for m in msgs if m.get("extra", {}).get("timestamp")]
    wall_clock = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0

    return {
        "instance_id": iid,
        "condition": "text",
        "steps": info.get("model_stats", {}).get("api_calls", 0),
        "cost": info.get("model_stats", {}).get("instance_cost", 0),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "wall_clock_s": round(wall_clock, 1),
        "patch_length": len(info.get("submission", "")),
        "exit_status": info.get("exit_status", ""),
        "n_images": 0,
        "obs_rendered": 0,
        "obs_total": 0,
    }


def extract_optical_instance(iid: str) -> dict:
    traj_file = OPTICAL_TRAJ_DIR / iid / f"{iid}.traj.json"
    if not traj_file.exists():
        return {"instance_id": iid, "condition": "optical", "error": "no trajectory"}

    traj = json.loads(traj_file.read_text())
    info = traj["info"]
    msgs = traj.get("messages", [])

    total_input = 0
    total_output = 0
    for m in msgs:
        usage = m.get("extra", {}).get("response", {}).get("usage", {})
        total_input += usage.get("prompt_tokens", 0)
        total_output += usage.get("completion_tokens", 0)

    return {
        "instance_id": iid,
        "condition": "optical",
        "steps": info.get("model_stats", {}).get("api_calls", 0),
        "cost": info.get("model_stats", {}).get("instance_cost", 0),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "wall_clock_s": info.get("wall_clock_time", 0),
        "patch_length": len(info.get("submission", "")),
        "exit_status": info.get("exit_status", ""),
        "n_images": info.get("total_images", 0) + info.get("ps_images", 0),
        "obs_rendered": info.get("observations_rendered", 0),
        "obs_total": info.get("observations_total", 0),
    }


def load_eval_results() -> tuple[dict, dict]:
    """Load SWE-bench evaluation results."""
    text_resolved = set()
    optical_resolved = set()

    for f in PROJECT_ROOT.rglob("*text_100*.json"):
        if f.stat().st_size > 100:
            try:
                rpt = json.loads(f.read_text())
                if "resolved_ids" in rpt:
                    text_resolved = set(rpt["resolved_ids"])
                    break
            except:
                pass

    for f in PROJECT_ROOT.rglob("*optical_100*.json"):
        if f.stat().st_size > 100:
            try:
                rpt = json.loads(f.read_text())
                if "resolved_ids" in rpt:
                    optical_resolved = set(rpt["resolved_ids"])
                    break
            except:
                pass

    return text_resolved, optical_resolved


def main():
    subset = [l.strip() for l in SUBSET_FILE.read_text().splitlines() if l.strip()]
    text_resolved, optical_resolved = load_eval_results()

    instances = []
    for iid in subset:
        t = extract_text_instance(iid)
        o = extract_optical_instance(iid)
        t["resolved"] = iid in text_resolved
        o["resolved"] = iid in optical_resolved
        instances.append(t)
        instances.append(o)

    # Compute summaries
    def summarize(items):
        valid = [i for i in items if "error" not in i]
        n = len(valid)
        if n == 0:
            return {}
        resolved = sum(1 for i in valid if i["resolved"])
        return {
            "n": n,
            "resolved": resolved,
            "resolve_rate": round(resolved / n, 4),
            "avg_steps": round(sum(i["steps"] for i in valid) / n, 1),
            "avg_cost": round(sum(i["cost"] for i in valid) / n, 4),
            "total_cost": round(sum(i["cost"] for i in valid), 2),
            "avg_input_tokens": round(sum(i["input_tokens"] for i in valid) / n, 0),
            "avg_output_tokens": round(sum(i["output_tokens"] for i in valid) / n, 0),
            "avg_wall_clock": round(sum(i["wall_clock_s"] for i in valid) / n, 1),
            "avg_patch_length": round(sum(i["patch_length"] for i in valid) / n, 0),
            "avg_images": round(sum(i["n_images"] for i in valid) / n, 1),
            "patches_with_content": sum(1 for i in valid if i["patch_length"] > 0),
            "limits_exceeded": sum(1 for i in valid if i["exit_status"] == "LimitsExceeded"),
        }

    text_items = [i for i in instances if i["condition"] == "text"]
    optical_items = [i for i in instances if i["condition"] == "optical"]

    summary = {
        "text": summarize(text_items),
        "optical": summarize(optical_items),
    }

    # Per-instance comparison
    comparison = []
    for iid in subset:
        t = next((i for i in text_items if i["instance_id"] == iid), {})
        o = next((i for i in optical_items if i["instance_id"] == iid), {})
        if "error" in t or "error" in o:
            continue
        comparison.append({
            "instance_id": iid,
            "text_resolved": t.get("resolved", False),
            "optical_resolved": o.get("resolved", False),
            "text_steps": t.get("steps", 0),
            "optical_steps": o.get("steps", 0),
            "text_cost": t.get("cost", 0),
            "optical_cost": o.get("cost", 0),
            "optical_images": o.get("n_images", 0),
            "optical_obs_rendered": o.get("obs_rendered", 0),
            "optical_obs_total": o.get("obs_total", 0),
        })

    # Instances where results differ
    diff = {
        "text_only": [c["instance_id"] for c in comparison if c["text_resolved"] and not c["optical_resolved"]],
        "optical_only": [c["instance_id"] for c in comparison if not c["text_resolved"] and c["optical_resolved"]],
        "both": [c["instance_id"] for c in comparison if c["text_resolved"] and c["optical_resolved"]],
        "neither": [c["instance_id"] for c in comparison if not c["text_resolved"] and not c["optical_resolved"]],
    }

    output = {
        "summary": summary,
        "instances": instances,
        "comparison": comparison,
        "diff": diff,
    }

    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"Saved to {OUTPUT}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'Metric':<25s} {'Text':>12s} {'Optical':>12s}")
    print(f"{'='*60}")
    for key in ["n", "resolved", "resolve_rate", "avg_steps", "avg_cost", "total_cost",
                 "avg_input_tokens", "avg_output_tokens", "avg_wall_clock", "avg_images",
                 "patches_with_content", "limits_exceeded"]:
        tv = summary["text"].get(key, "—")
        ov = summary["optical"].get(key, "—")
        if isinstance(tv, float):
            tv = f"{tv:.4f}" if key == "resolve_rate" else f"{tv:.2f}" if "cost" in key else f"{tv:.1f}"
        if isinstance(ov, float):
            ov = f"{ov:.4f}" if key == "resolve_rate" else f"{ov:.2f}" if "cost" in key else f"{ov:.1f}"
        print(f"{key:<25s} {str(tv):>12s} {str(ov):>12s}")

    print(f"\nText-only resolved: {diff['text_only']}")
    print(f"Optical-only resolved: {diff['optical_only']}")
    print(f"Both resolved: {len(diff['both'])}")
    print(f"Neither resolved: {len(diff['neither'])}")


if __name__ == "__main__":
    main()
