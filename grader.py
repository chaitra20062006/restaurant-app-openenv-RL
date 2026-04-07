"""
grader.py
Deterministic, reproducible grader for the Restaurant Table Management environment.

Evaluates three dimensions:
  1. Efficiency  – table utilisation and best-fit seating rate
  2. Revenue     – total revenue vs theoretical maximum
  3. Satisfaction – seated customers vs all arrivals (including walkouts)

Final score is a weighted harmonic mean strictly in (0.0, 1.0).
Scores are NEVER constant – they depend on episode statistics.

Usage:
  python grader.py --task easy
  python grader.py --task medium --runs 3
  python grader.py --base-url http://localhost:8000 --task hard
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import random
import logging
from typing import Dict, Any, List

import requests

from models import GraderResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] grader: %(message)s",
)
logger = logging.getLogger("grader")

# ---------------------------------------------------------------------------
# Task-specific theoretical maximums (pre-computed, deterministic)
# ---------------------------------------------------------------------------
TASK_MAX_REVENUE: Dict[str, float] = {
    "easy":   3_500.0,
    "medium": 8_000.0,
    "hard":  18_000.0,
}

TASK_MAX_SEATED: Dict[str, int] = {
    "easy":   30,
    "medium": 70,
    "hard":  140,
}

WEIGHTS = {"efficiency": 0.35, "revenue": 0.35, "satisfaction": 0.30}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _sigmoid_score(raw: float, midpoint: float = 0.5, steepness: float = 8.0) -> float:
    """Map any float to (0, 1) with a sigmoid curve. Never hits 0 or 1 exactly."""
    x = steepness * (raw - midpoint)
    s = 1.0 / (1.0 + math.exp(-x))
    # Clip away from extremes: [0.02, 0.98]
    return max(0.02, min(0.98, s))


def compute_efficiency_score(obs: Dict[str, Any], task: str) -> float:
    """
    Efficiency = fraction of occupied table-steps over total table-steps.
    A perfect episode keeps all tables occupied at all times.
    """
    tables = obs.get("tables", [])
    if not tables:
        return 0.1
    occupied = sum(1 for t in tables if t.get("status") == "occupied")
    total = len(tables)
    raw = occupied / total if total > 0 else 0.0
    return _sigmoid_score(raw, midpoint=0.45)


def compute_revenue_score(total_revenue: float, task: str) -> float:
    """Revenue score normalised to task theoretical maximum."""
    max_rev = TASK_MAX_REVENUE.get(task, 5000.0)
    raw = min(total_revenue / max_rev, 1.0)
    return _sigmoid_score(raw, midpoint=0.40)


def compute_satisfaction_score(
    total_seated: int,
    total_rejected: int,
    total_walkouts: int,
) -> float:
    """Satisfaction = seated / (seated + rejected + walkouts). Penalises all losses."""
    total_arrivals = total_seated + total_rejected + total_walkouts
    if total_arrivals == 0:
        return 0.1  # no customers ever came – bad for grader
    raw = total_seated / total_arrivals
    return _sigmoid_score(raw, midpoint=0.55)


def weighted_harmonic_mean(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Weighted harmonic mean – penalises low scores heavily.
    Returns value in (0.0, 1.0) since all inputs are in (0.02, 0.98).
    """
    numerator = sum(weights[k] for k in scores)
    denominator = sum(weights[k] / scores[k] for k in scores)
    if denominator == 0:
        return 0.02
    return max(0.02, min(0.98, numerator / denominator))


# ---------------------------------------------------------------------------
# Main grading logic
# ---------------------------------------------------------------------------

def grade_episode(
    base_url: str,
    task: str,
    seed: int,
    max_steps: int = 250,
) -> GraderResult:
    """Run one episode with a simple heuristic policy and grade the result."""

    # 1. Reset
    resp = requests.post(f"{base_url}/reset", json={"task": task, "seed": seed}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observation"]

    logger.info("[START] grading task=%s seed=%d", task, seed)

    total_reward = 0.0
    done = False
    step_count = 0
    ACTIONS = ["assign_table", "assign_table", "assign_table", "delay_seating", "reject_customer"]

    while not done and step_count < max_steps:
        # Deterministic heuristic policy:
        #   - If queue non-empty and table available → assign
        #   - If queue very long and stuck → reject occasionally
        #   - Else delay
        queue = obs.get("waiting_queue", [])
        tables = obs.get("tables", [])
        available = [t for t in tables if t.get("status") == "available"]

        if queue and available:
            party = queue[0]["party_size"]
            fit = [t for t in available if t["capacity"] >= party]
            if fit:
                action = "assign_table"
            elif len(queue) > 5:
                action = "reject_customer"
            else:
                action = "delay_seating"
        elif not queue:
            action = "delay_seating"
        else:
            # Queue but no fit – try combine if two smalls available
            smalls = [t for t in available if t["capacity"] <= 4]
            if len(smalls) >= 2 and queue:
                payload = {
                    "action": "combine_tables",
                    "table_id": smalls[0]["id"],
                    "combine_with": smalls[1]["id"],
                }
            else:
                action = "delay_seating"
                payload = {"action": action}
            step_resp = requests.post(f"{base_url}/step", json=payload, timeout=30)
            step_resp.raise_for_status()
            data = step_resp.json()
            obs = data["observation"]
            total_reward += data["reward"]
            done = data["done"]
            step_count += 1
            logger.info("[STEP] t=%d action=combine reward=%.2f done=%s",
                        step_count, data["reward"], done)
            continue

        payload = {"action": action}
        step_resp = requests.post(f"{base_url}/step", json=payload, timeout=30)
        step_resp.raise_for_status()
        data = step_resp.json()
        obs = data["observation"]
        total_reward += data["reward"]
        done = data["done"]
        step_count += 1
        logger.info("[STEP] t=%d action=%s reward=%.2f done=%s",
                    step_count, action, data["reward"], done)

    # 2. Compute sub-scores
    eff   = compute_efficiency_score(obs, task)
    rev   = compute_revenue_score(obs.get("total_revenue", 0.0), task)
    sat   = compute_satisfaction_score(
                obs.get("total_seated", 0),
                obs.get("total_rejected", 0),
                obs.get("total_walkouts", 0),
            )
    scores = {"efficiency": eff, "revenue": rev, "satisfaction": sat}
    final  = weighted_harmonic_mean(scores, WEIGHTS)

    result = GraderResult(
        score=round(final, 4),
        efficiency_score=round(eff, 4),
        revenue_score=round(rev, 4),
        satisfaction_score=round(sat, 4),
        details={
            "task": task,
            "seed": seed,
            "steps": step_count,
            "total_reward": round(total_reward, 4),
            "total_revenue": obs.get("total_revenue", 0.0),
            "total_seated": obs.get("total_seated", 0),
            "total_rejected": obs.get("total_rejected", 0),
            "total_walkouts": obs.get("total_walkouts", 0),
        },
    )

    logger.info(
        "[END] task=%s score=%.4f eff=%.4f rev=%.4f sat=%.4f",
        task, final, eff, rev, sat,
    )
    return result


def run_all_tasks(base_url: str, runs: int = 1) -> Dict[str, Any]:
    """Grade all three tasks and aggregate results."""
    results: Dict[str, Any] = {}
    for task in ["easy", "medium", "hard"]:
        task_scores: List[float] = []
        for run in range(runs):
            seed = 42 + run * 17  # deterministic, distinct seeds per run
            gr = grade_episode(base_url, task, seed)
            task_scores.append(gr.score)
            results[f"{task}_run{run}"] = gr.dict()

        avg = sum(task_scores) / len(task_scores)
        results[f"{task}_avg_score"] = round(avg, 4)

    overall = sum(
        results[f"{t}_avg_score"] for t in ["easy", "medium", "hard"]
    ) / 3
    results["overall_score"] = round(overall, 4)
    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv Restaurant Grader")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.task == "all":
        results = run_all_tasks(args.base_url, runs=args.runs)
        print(json.dumps(results, indent=2))
    else:
        seed = args.seed if args.seed is not None else 42
        gr = grade_episode(args.base_url, args.task, seed)
        print(json.dumps(gr.dict(), indent=2))
