"""Run SWE-bench experiment: Text vs Optical (observation-level).

Usage:
  python scripts/run_experiment.py --condition text --n 100 -w 32
  python scripts/run_experiment.py --condition optical --n 100 -w 32
  python scripts/run_experiment.py --condition text --n 1 --filter "django__django-15987"
"""

import json
import subprocess
import sys
import time
import concurrent.futures
import traceback
from pathlib import Path

import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from optical_agent import OpticalAgent
from optical_preprocessor import preprocess_optical_file, preprocess_text

app = typer.Typer()

# mini-swe-agent's built-in swebench config (installed via pip)
import minisweagent
CONFIG_TEXT = str(Path(minisweagent.__file__).parent / "config" / "benchmarks" / "swebench.yaml")
CONFIG_OPTICAL = str(PROJECT_ROOT / "src" / "config_optical.yaml")
SUBSET_FILE = PROJECT_ROOT / "data" / "subset_100.txt"

import threading
_PREDS_LOCK = threading.Lock()


def update_preds_file(preds_file: Path, instance_id: str, model_name: str, result: str):
    with _PREDS_LOCK:
        data = {}
        if preds_file.exists():
            data = json.loads(preds_file.read_text())
        data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        preds_file.write_text(json.dumps(data, indent=2))


def get_instance_ids(n: int, filter_id: str = "") -> list[str]:
    if filter_id:
        return [filter_id]
    if SUBSET_FILE.exists():
        ids = [l.strip() for l in SUBSET_FILE.read_text().splitlines() if l.strip()]
        return ids[:n]
    raise FileNotFoundError(f"Subset file not found: {SUBSET_FILE}")


def process_optical_instance(iid, instance, config, output_dir):
    """Process one instance under the optical condition."""
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    preds_file = output_dir / "preds.json"

    # Preprocess problem_statement (code blocks → images)
    t_start = time.time()
    task = preprocess_optical_file(instance["problem_statement"])
    n_ps_images = task.count("MSWEA_MULTIMODAL_CONTENT") // 2

    model = get_model(config=config.get("model", {}))
    agent = None
    exit_status = None
    result = None

    try:
        env = get_sb_environment(config, instance)
        agent = OpticalAgent(model, env, **config.get("agent", {}))
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")
        wall_time = time.time() - t_start

        stats = agent.render_stats
        print(f"  [{iid}] Exit: {exit_status}, Steps: {agent.n_calls}, Cost: ${agent.cost:.2f}, "
              f"Obs rendered: {stats['observations_rendered']}/{stats['observations_total']}, "
              f"Images: {stats['total_images']+n_ps_images}")
    except Exception as e:
        print(f"  [{iid}] ERROR: {e}")
        exit_status = type(e).__name__
        result = ""
        wall_time = time.time() - t_start
        traceback.print_exc()
    finally:
        if agent:
            traj_path = output_dir / iid / f"{iid}.traj.json"
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(traj_path, {
                "info": {
                    "exit_status": exit_status,
                    "submission": result,
                    "condition": "optical",
                    "wall_clock_time": round(wall_time, 2),
                    "ps_images": n_ps_images,
                    **agent.render_stats,
                },
                "instance_id": iid,
            })
        update_preds_file(preds_file, iid, model.config.model_name, result or "")


@app.command()
def main(
    condition: str = typer.Option("text", help="Condition: text or optical"),
    n: int = typer.Option(20, help="Number of instances"),
    filter_id: str = typer.Option("", "--filter", help="Run only this instance ID"),
    workers: int = typer.Option(128, "-w", help="Parallel workers"),
    output_suffix: str = typer.Option("", "--suffix", help="Suffix for output dir, e.g. '_100'"),
):
    output_dir = PROJECT_ROOT / "results" / f"{condition}{output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_ids = get_instance_ids(n, filter_id)
    print(f"Condition: {condition}")
    print(f"Instances: {len(instance_ids)}")
    print(f"Workers: {workers}")
    print(f"Output: {output_dir}")

    # Skip already done
    preds_file = output_dir / "preds.json"
    done_ids = set()
    if preds_file.exists():
        done_ids = set(json.loads(preds_file.read_text()).keys())

    todo_ids = [iid for iid in instance_ids if iid not in done_ids]
    print(f"TODO: {len(todo_ids)} (skipping {len(done_ids)} done)")

    if condition == "text":
        # Use standard swebench CLI with GPT-5-mini
        filter_regex = "|".join(todo_ids)
        if not todo_ids:
            print("All done!")
            return
        cmd = [
            "python", "-m", "minisweagent.run.benchmarks.swebench",
            "--subset", "verified",
            "--split", "test",
            "--filter", filter_regex,
            "-o", str(output_dir),
            "-c", CONFIG_TEXT,
            "-c", "model.model_name=openai/gpt-5-mini-2025-08-07",
            "-c", "model.cost_tracking=ignore_errors",
            "-c", "model.model_kwargs.drop_params=true",
            "-w", str(workers),
        ]
        print(f"Running mini-swe-agent CLI with {len(todo_ids)} instances...")
        subprocess.run(cmd, cwd=str(REPO_ROOT))

    elif condition == "optical":
        from datasets import load_dataset
        from minisweagent.config import get_config_from_spec

        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        instances = {ex["instance_id"]: ex for ex in ds}
        config = get_config_from_spec(CONFIG_OPTICAL)

        args_list = []
        for iid in todo_ids:
            if iid in instances:
                args_list.append((iid, instances[iid], config, output_dir))

        print(f"Running {len(args_list)} instances with {min(workers, len(args_list))} workers...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_optical_instance, *args): args[0] for args in args_list}
            for future in concurrent.futures.as_completed(futures):
                iid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  [{iid}] UNCAUGHT: {e}")
    else:
        print(f"Unknown condition: {condition}")
        raise typer.Exit(1)

    # Print summary
    if preds_file.exists():
        preds = json.loads(preds_file.read_text())
        n_patches = sum(1 for v in preds.values() if len(v.get("model_patch", "")) > 0)
        print(f"\nDone: {len(preds)} instances, {n_patches} with patches")


if __name__ == "__main__":
    app()
