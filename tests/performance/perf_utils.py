"""
Simple utilities for running performance tests.
"""

import time
import statistics
from typing import Callable, List, Dict, Any

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def run_perf_test(func: Callable, rounds: int, *args, **kwargs) -> Dict[str, float]:
    """
    Run a function multiple times and collect timing statistics.

    Args:
        func: Function to test
        rounds: Number of times to run
        *args, **kwargs: Arguments to pass to func

    Returns:
        Dict with timing statistics in seconds (min, max, mean, median, stdev)
    """
    times = []

    iterator = tqdm(range(rounds), desc="  Rounds", leave=False, unit="round") if HAS_TQDM else range(rounds)

    for _ in iterator:
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed_s = time.perf_counter() - start
        times.append(elapsed_s)

    return {
        "min": min(times),
        "max": max(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "rounds": rounds
    }


def print_perf_table(results: List[Dict[str, Any]]):
    """
    Print performance results in a tabular format.

    Args:
        results: List of dicts with keys: name, min, max, mean, median, stdev, rounds (all times in seconds)
    """
    print("\n" + "=" * 90)
    print(f"{'Test Name':<30} {'Min (s)':<12} {'Max (s)':<12} {'Mean (s)':<12} {'Median (s)':<12}")
    print("=" * 90)

    for r in results:
        print(
            f"{r['name']:<30} "
            f"{r['min']:>10.4f}   "
            f"{r['max']:>10.4f}   "
            f"{r['mean']:>10.4f}   "
            f"{r['median']:>10.4f}"
        )

    print("=" * 90)
    print(f"Rounds per test: {results[0]['rounds']}\n")
