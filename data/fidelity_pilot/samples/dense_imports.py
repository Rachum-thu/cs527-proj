"""Configuration module for the analysis pipeline."""

from __future__ import annotations

import os
import sys
import json
import logging
import hashlib
import functools
from pathlib import Path
from typing import Any, Optional, Union, Protocol
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""
    input_path: Path
    output_path: Path
    batch_size: int = 32
    max_workers: int = 4
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    features: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    verbose: bool = False

    def __post_init__(self):
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        import yaml
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)
