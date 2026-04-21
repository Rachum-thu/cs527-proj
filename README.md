# Optical Compression for Agentic Code Understanding

**CS 527 Research Project Report — Group 9**

Runchu Tian, Vikas Reddy | University of Illinois at Urbana-Champaign

## Overview

This repository contains the code, data, and paper for our study on **optical compression** — rendering source code as monospace images — as a drop-in replacement for text-based code input in an agentic software engineering pipeline.

We evaluate GPT-5-mini on a 100-instance subset of SWE-bench Verified using the [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) framework, comparing a standard text-based agent against an optical variant that renders code-heavy tool outputs as images.

### Key Findings

- **Feasibility without post-training**: GPT-5-mini reads Python code from rendered images with ≤1.5% character error rate and 100% identifier recall. On SWE-bench Verified, the optical condition matches the text baseline exactly (51% resolve rate each).
- **Cost overhead**: Image tokens are currently more expensive than text tokens — the optical agent consumes 29% more input tokens and costs 1.9× more per instance. Post-training (e.g., trained visual encoders) would be needed to make optical compression cost-competitive.

## Repository Structure

```
├── paper/                      # LaTeX source and compiled PDF
│   ├── main.tex
│   ├── main.pdf                # Compiled paper
│   └── Sections/               # Per-section .tex files
├── src/                        # Core implementation
│   ├── optical_agent.py        # OpticalAgent — renders code tool outputs as images
│   ├── optical_preprocessor.py # Preprocesses problem statements (code blocks → images)
│   ├── render_code_to_image.py # Text → PNG rendering (Pillow-based)
│   └── config_optical.yaml     # mini-swe-agent config for optical condition
├── scripts/                    # Experiment scripts
│   ├── run_experiment.py       # Main experiment runner (text / optical)
│   ├── run_fidelity_pilot.py   # Image readability pilot study
│   ├── build_subset_100.py     # Construct stratified 100-instance subset
│   ├── aggregate_results.py    # Aggregate results into paper_data.json
│   └── plot_figures.py         # Generate paper figures (matplotlib)
├── data/                       # Experiment definitions and pilot data
│   ├── subset_100.txt          # 100 instance IDs (one per line)
│   └── fidelity_pilot/         # Readability pilot samples and results
├── results/                    # Experiment results
│   ├── text/preds.json         # Text condition predictions
│   ├── optical/preds.json      # Optical condition predictions
│   ├── eval/                   # SWE-bench evaluation reports
│   └── paper_data.json         # Aggregated data used in the paper
└── README.md
```

## Trajectory Data

Full agent trajectories (.traj.json files) are large and hosted separately:

**HuggingFace**: [autoprogrammer/cs527-optical-compression-trajectories](https://huggingface.co/datasets/autoprogrammer/cs527-optical-compression-trajectories)

These contain the complete message history for each instance × condition, including all tool calls, observations, and model responses.

## Setup

### Prerequisites

- Python 3.10+
- Docker (for SWE-bench evaluation)
- OpenAI API key with GPT-5-mini access

### Installation

```bash
# Install mini-swe-agent
pip install mini-swe-agent[full]

# Install additional dependencies
pip install Pillow python-Levenshtein jiwer matplotlib
```

## Reproducing Results

### Step 1: Build the 100-instance subset

```bash
python scripts/build_subset_100.py
```

### Step 2: Run the image readability pilot

```bash
python scripts/run_fidelity_pilot.py
```

### Step 3: Run the text baseline

```bash
python scripts/run_experiment.py --condition text --n 100 -w 32
```

### Step 4: Run the optical condition

```bash
python scripts/run_experiment.py --condition optical --n 100 -w 32
```

### Step 5: Run SWE-bench evaluation

```bash
# Text
python -m swebench.harness.run_evaluation \
    -d princeton-nlp/SWE-bench_Verified -s test \
    -p results/text/swebench_preds.json \
    --run_id text_100 --max_workers 16

# Optical
python -m swebench.harness.run_evaluation \
    -d princeton-nlp/SWE-bench_Verified -s test \
    -p results/optical/swebench_preds.json \
    --run_id optical_100 --max_workers 16
```

### Step 6: Aggregate results and generate figures

```bash
python scripts/aggregate_results.py
python scripts/plot_figures.py
```

## Paper

The compiled paper is at `paper/main.pdf`. To recompile:

```bash
cd paper && tectonic main.tex
```

## Citation

```bibtex
@misc{tian2026optical,
  title={Optical Compression for Agentic Code Understanding},
  author={Tian, Runchu and Reddy, Vikas},
  year={2026},
  note={CS 527 Course Project, UIUC}
}
```
