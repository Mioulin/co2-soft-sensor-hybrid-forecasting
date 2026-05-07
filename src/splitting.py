"""
Whole-run train/val/test split.

No random row split: overlapping sliding windows would leak temporal
information across the boundary. Each run is used entirely in one split.
"""

from typing import Dict, List, Tuple
import numpy as np
from src.config import TRAIN_RUNS, VAL_RUNS, TEST_RUNS


def split_runs(
    runs: Dict[str, dict],
    train_ids: List[str] = TRAIN_RUNS,
    val_ids:   List[str] = VAL_RUNS,
    test_ids:  List[str] = TEST_RUNS,
) -> Tuple[Dict, Dict, Dict]:
    """
    Partition loaded runs dict into train / val / test subsets.
    """
    def _subset(ids):
        missing = [r for r in ids if r not in runs]
        if missing:
            raise KeyError(f"Runs not loaded: {missing}")
        return {r: runs[r] for r in ids}

    return _subset(train_ids), _subset(val_ids), _subset(test_ids)
