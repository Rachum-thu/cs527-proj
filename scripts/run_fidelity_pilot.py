"""Fidelity v2: iterative evaluation of GPT-5-mini code image reading.

Tests with actual Python source code (not diffs).
Runs one setting at a time for quick iteration.
"""

import base64
import json
import time
import sys
from pathlib import Path

import Levenshtein
from openai import OpenAI

SAMPLES_DIR = Path(__file__).parent / "samples"
IMAGES_DIR = Path(__file__).parent / "images"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from render_code_to_image import render_and_save

client = OpenAI()
MODEL = "gpt-5-mini-2025-08-07"

TRANSCRIPTION_PROMPT = (
    "Transcribe the exact text content shown in this image. "
    "Preserve all whitespace, indentation, symbols, and formatting exactly as shown. "
    "Do not add any explanation or commentary — output ONLY the transcribed text."
)

SAMPLE_NAMES = [
    "agents_default_short",    # 50 lines, real code
    "agents_default_medium",   # 100 lines, real code
    "agents_default_full",     # 156 lines, real code
    "dense_imports",           # 47 lines, imports heavy
    "complex_logic",           # 97 lines, nested logic
    "special_chars",           # 59 lines, regex/special chars
]


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def transcribe_images(image_paths: list[Path]) -> tuple[str, dict]:
    content = [{"type": "text", "text": TRANSCRIPTION_PROMPT}]
    for p in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(p)}"},
        })

    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=16384,
        messages=[{"role": "user", "content": content}],
    )
    elapsed = time.time() - t0

    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "wall_clock_s": round(elapsed, 2),
    }
    return text, usage


def compute_fidelity(ground_truth: str, transcription: str) -> dict:
    gt = ground_truth.strip()
    tr = transcription.strip()

    # Remove header line if model included it
    tr_lines_raw = tr.split("\n")
    if tr_lines_raw and ("—" in tr_lines_raw[0] or "part" in tr_lines_raw[0].lower()):
        tr = "\n".join(tr_lines_raw[1:]).strip()

    # CER
    cer = Levenshtein.distance(gt, tr) / max(len(gt), 1)

    # Line-level exact match
    gt_lines = gt.split("\n")
    tr_lines = tr.split("\n")
    max_lines = max(len(gt_lines), len(tr_lines))
    matches = sum(1 for g, t in zip(gt_lines, tr_lines) if g == t)
    strict_line_em = matches / max(max_lines, 1)

    # Stripped line match (ignore leading/trailing whitespace)
    stripped_matches = sum(1 for g, t in zip(gt_lines, tr_lines) if g.strip() == t.strip())
    stripped_line_em = stripped_matches / max(max_lines, 1)

    # Indentation accuracy: for matching stripped lines, is indentation correct?
    indent_correct = 0
    indent_total = 0
    for g, t in zip(gt_lines, tr_lines):
        if g.strip() == t.strip() and g.strip():  # content matches
            indent_total += 1
            g_indent = len(g) - len(g.lstrip())
            t_indent = len(t) - len(t.lstrip())
            if g_indent == t_indent:
                indent_correct += 1
    indent_accuracy = indent_correct / max(indent_total, 1)

    # Identifier preservation: extract Python identifiers and compare
    import re
    gt_ids = set(re.findall(r'\b[a-zA-Z_]\w*\b', gt))
    tr_ids = set(re.findall(r'\b[a-zA-Z_]\w*\b', tr))
    id_recall = len(gt_ids & tr_ids) / max(len(gt_ids), 1)

    # String literal preservation
    gt_strings = set(re.findall(r'["\']([^"\']{3,})["\']', gt))
    tr_strings = set(re.findall(r'["\']([^"\']{3,})["\']', tr))
    string_recall = len(gt_strings & tr_strings) / max(len(gt_strings), 1)

    return {
        "cer": round(cer, 4),
        "strict_line_em": round(strict_line_em, 4),
        "stripped_line_em": round(stripped_line_em, 4),
        "indent_accuracy": round(indent_accuracy, 4),
        "identifier_recall": round(id_recall, 4),
        "string_recall": round(string_recall, 4),
        "gt_lines": len(gt_lines),
        "tr_lines": len(tr_lines),
    }


def run_setting(setting_name: str, font_size: int, page_width_chars: int = 100,
                lines_per_image: int = 80):
    print(f"\n{'='*70}")
    print(f"Setting: {setting_name} (font={font_size}, width={page_width_chars}, lines/img={lines_per_image})")
    print(f"{'='*70}")

    results = []
    for sample_name in SAMPLE_NAMES:
        sample_path = SAMPLES_DIR / f"{sample_name}.py"
        if not sample_path.exists():
            print(f"  SKIP {sample_name}")
            continue

        gt_text = sample_path.read_text()
        meta = json.loads((SAMPLES_DIR / f"{sample_name}.json").read_text())

        # Render
        out_dir = IMAGES_DIR / setting_name
        img_paths = render_and_save(
            gt_text, out_dir, sample_name,
            font_size=font_size, page_width_chars=page_width_chars,
            lines_per_image=lines_per_image, header=f"{sample_name}.py",
        )
        n_images = len(img_paths)
        total_kb = sum(p.stat().st_size for p in img_paths) / 1024

        print(f"  {sample_name} ({meta['lines']}L, {n_images} img, {total_kb:.0f}KB) ... ", end="", flush=True)

        try:
            transcription, usage = transcribe_images(img_paths)
            fidelity = compute_fidelity(gt_text, transcription)

            result = {
                "setting": setting_name,
                "sample": sample_name,
                "gt_lines": meta["lines"],
                "n_images": n_images,
                "image_kb": round(total_kb, 1),
                **usage,
                **fidelity,
            }
            results.append(result)

            # Save transcription
            trans_dir = RESULTS_DIR / "transcriptions" / setting_name
            trans_dir.mkdir(parents=True, exist_ok=True)
            (trans_dir / f"{sample_name}.txt").write_text(transcription)

            print(f"CER={fidelity['cer']:.3f}  stripEM={fidelity['stripped_line_em']:.2f}  "
                  f"indent={fidelity['indent_accuracy']:.2f}  idRecall={fidelity['identifier_recall']:.2f}  "
                  f"tok={usage['input_tokens']}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"setting": setting_name, "sample": sample_name, "error": str(e)})

    return results


def main():
    settings = [
        ("f20", 20, 100, 80),
        ("f18", 18, 100, 80),
        ("f16", 16, 100, 80),
        ("f14", 14, 100, 80),
        ("f12", 12, 100, 80),
        ("f10", 10, 100, 80),
        ("f8",   8, 100, 80),
    ]

    # Allow running a single setting via CLI arg
    if len(sys.argv) > 1:
        target = sys.argv[1]
        settings = [(n, fs, pw, lpi) for n, fs, pw, lpi in settings if n == target]
        if not settings:
            print(f"Unknown setting: {target}")
            sys.exit(1)

    all_results = []
    for name, fs, pw, lpi in settings:
        results = run_setting(name, fs, pw, lpi)
        all_results.extend(results)

    # Save
    out_file = RESULTS_DIR / "fidelity_v2.json"
    # Merge with existing if present
    if out_file.exists():
        existing = json.loads(out_file.read_text())
        # Remove old results for settings we just ran
        ran_settings = {r["setting"] for r in all_results if "error" not in r}
        existing = [r for r in existing if r.get("setting") not in ran_settings]
        all_results = existing + all_results
    out_file.write_text(json.dumps(all_results, indent=2))

    # Print summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    valid = [r for r in all_results if "error" not in r]
    print(f"{'Setting':<10s} {'Sample':<25s} {'Lines':>5s} {'Imgs':>4s} {'CER':>6s} {'StripEM':>7s} "
          f"{'Indent':>6s} {'IdRec':>5s} {'InTok':>6s}")
    print("-" * 90)
    for r in sorted(valid, key=lambda x: (x["setting"], x["sample"])):
        print(f"{r['setting']:<10s} {r['sample']:<25s} {r['gt_lines']:>5d} {r['n_images']:>4d} "
              f"{r['cer']:>6.3f} {r['stripped_line_em']:>7.2f} {r['indent_accuracy']:>6.2f} "
              f"{r['identifier_recall']:>5.2f} {r['input_tokens']:>6d}")

    # Per-setting averages
    from collections import defaultdict
    by_setting = defaultdict(list)
    for r in valid:
        by_setting[r["setting"]].append(r)

    print(f"\n{'Setting':<10s} {'AvgCER':>7s} {'AvgStripEM':>10s} {'AvgIndent':>9s} {'AvgIdRec':>8s} {'AvgTok':>7s}")
    print("-" * 60)
    for s in sorted(by_setting.keys()):
        items = by_setting[s]
        print(f"{s:<10s} {sum(r['cer'] for r in items)/len(items):>7.3f} "
              f"{sum(r['stripped_line_em'] for r in items)/len(items):>10.2f} "
              f"{sum(r['indent_accuracy'] for r in items)/len(items):>9.2f} "
              f"{sum(r['identifier_recall'] for r in items)/len(items):>8.2f} "
              f"{sum(r['input_tokens'] for r in items)/len(items):>7.0f}")


if __name__ == "__main__":
    main()
