"""
server/environment.py
Core Restaurant Table Management RL Environment.

Implements the OpenEnv loop:  observe → act → reward → repeat

Endpoints exposed (consumed by app.py):
  reset(task, seed)  → ObservationState
  step(action, ...)  → (ObservationState, reward, done, info)
  state()            → ObservationState
"""
from __future__ import annotations

import random
import math
import logging
from copy import deepcopy
from typing import Optional, Dict, Any, Tuple, List

# Sibling import – works whether run from repo root or via uvicorn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import (
    Table, Customer, ObservationState,
    TableStatus, Action, TaskDifficulty,
)

logger = logging.getLogger("restaurant_env")

# ---------------------------------------------------------------------------
# Task configurations (mirror openenv.yaml)
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "num_tables": 6,
        "table_sizes": [2, 2, 4, 4, 6, 8],
        "arrival_rate": 0.30,
        "max_patience": 8,
        "max_steps": 100,
        "seed": 42,
    },
    "medium": {
        "num_tables": 10,
        "table_sizes": [2, 2, 2, 4, 4, 4, 6, 6, 8, 10],
        "arrival_rate": 0.55,
        "max_patience": 5,
        "max_steps": 150,
        "seed": 123,
    },
    "hard": {
        "num_tables": 14,
        "table_sizes": [2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 8, 8, 10],
        "arrival_rate": 0.80,
        "max_patience": 3,
        "max_steps": 200,
        "seed": 999,
    },
}

# Revenue per seated person (varies with party size efficiency)
BASE_REVENUE_PER_PERSON = 25.0
# Avg dining duration in steps
MIN_DINING_STEPS = 4
MAX_DINING_STEPS = 12


class RestaurantEnv:
    """
    Stateful RL environment for restaurant table management.

    Observation space:
      - tables: List[Table]  (capacity, status, time_seated, party_size)
      - waiting_queue: List[Customer]  (party_size, patience_remaining)
      - time_step: int
      - occupancy_rate: float
      - total_seated, total_rejected, total_walkouts, total_revenue

    Action space (discrete, 4 actions):
      0  assign_table    – seat next customer in queue at best-fit table
      1  reject_customer – remove first queued customer (penalty)
      2  delay_seating   – do nothing (small penalty)
      3  combine_tables  – merge two adjacent small tables for large party

    Reward shaping:
      +  seated:         +10 base + utilization bonus
      +  utilization:    continuous bonus each step for high occupancy
      -  walkout:        -8 per customer who leaves due to impatience
      -  reject:         -5 per manual rejection
      -  idle tables:    -0.1 per unused table per step (only when queue > 0)
      -  delay:          -0.5 per delay action
      Episode ends at max_steps or if all customers served and queue empty.
    """

    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}
        self._rng: random.Random = random.Random()
        self._tables: List[Table] = []
        self._queue: List[Customer] = []
        self._time_step: int = 0
        self._max_steps: int = 100
        self._total_seated: int = 0
        self._total_rejected: int = 0
        self._total_walkouts: int = 0
        self._total_revenue: float = 0.0
        self._episode_active: bool = False
        self._customer_counter: int = 0
        # Dining timers: table_id → steps_remaining
        self._dining_timers: Dict[int, int] = {}
        self._task_name: str = "easy"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task: str = "easy", seed: Optional[int] = None) -> ObservationState:
        """Reset environment to initial state for given task difficulty."""
        self._cfg = deepcopy(TASK_CONFIGS[task])
        self._task_name = task
        actual_seed = seed if seed is not None else self._cfg["seed"]
        self._rng = random.Random(actual_seed)

        self._max_steps = self._cfg["max_steps"]
        self._time_step = 0
        self._total_seated = 0
        self._total_rejected = 0
        self._total_walkouts = 0
        self._total_revenue = 0.0
        self._episode_active = True
        self._customer_counter = 0
        self._dining_timers = {}

        # Build tables
        self._tables = [
            Table(id=i, capacity=cap, status=TableStatus.AVAILABLE)
            for i, cap in enumerate(self._cfg["table_sizes"])
        ]
        self._queue = []

        logger.info("[START] task=%s seed=%d max_steps=%d tables=%d",
                    task, actual_seed, self._max_steps, len(self._tables))
        return self._make_obs()

    def step(
        self,
        action: str,
        table_id: Optional[int] = None,
        combine_with: Optional[int] = None,
    ) -> Tuple[ObservationState, float, bool, Dict[str, Any]]:
        """
        Advance environment one step.
        Returns (observation, reward, done, info).
        """
        if not self._episode_active:
            raise RuntimeError("Environment not active. Call reset() first.")

        reward = 0.0
        info: Dict[str, Any] = {"action": action, "step": self._time_step}

        # 1. Tick dining timers (tables become free when timer hits 0)
        freed = self._tick_dining_timers()
        info["tables_freed"] = freed

        # 2. Spawn new customers probabilistically
        arrived = self._spawn_customers()
        info["arrived"] = arrived

        # 3. Process walkouts before agent acts
        walkouts = self._process_walkouts()
        reward -= walkouts * 8.0
        info["walkouts"] = walkouts

        # 4. Execute agent action
        action_reward, action_info = self._execute_action(action, table_id, combine_with)
        reward += action_reward
        info.update(action_info)

        # 5. Occupancy utilisation bonus (continuous shaping)
        occ = self._occupancy_rate()
        reward += occ * 2.0  # up to +2 per step for full house

        # 6. Idle penalty – when queue non-empty but tables sit empty
        if len(self._queue) > 0:
            idle = sum(1 for t in self._tables
                       if t.status == TableStatus.AVAILABLE
                       and t.capacity >= self._queue[0].party_size)
            reward -= idle * 0.1

        # 7. Advance time
        self._time_step += 1

        # 8. Check episode end
        done = self._time_step >= self._max_steps
        if done:
            self._episode_active = False
            # Final reward: bonus for low rejection rate
            if self._total_seated + self._total_rejected > 0:
                sat = self._total_seated / (self._total_seated + self._total_rejected + self._total_walkouts + 1e-9)
                reward += sat * 10.0
            logger.info(
                "[END] steps=%d seated=%d rejected=%d walkouts=%d revenue=%.1f",
                self._time_step, self._total_seated, self._total_rejected,
                self._total_walkouts, self._total_revenue,
            )

        obs = self._make_obs(done)
        logger.info(
            "[STEP] t=%d action=%s reward=%.2f occ=%.2f queue=%d done=%s",
            self._time_step, action, reward, occ, len(self._queue), done,
        )
        return obs, round(reward, 4), done, info

    def state(self) -> ObservationState:
        """Return current environment observation without advancing step."""
        return self._make_obs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs(self, done: bool = False) -> ObservationState:
        return ObservationState(
            tables=deepcopy(self._tables),
            waiting_queue=deepcopy(self._queue),
            time_step=self._time_step,
            occupancy_rate=round(self._occupancy_rate(), 4),
            total_seated=self._total_seated,
            total_rejected=self._total_rejected,
            total_walkouts=self._total_walkouts,
            total_revenue=round(self._total_revenue, 2),
            episode_done=done,
        )

    def _occupancy_rate(self) -> float:
        if not self._tables:
            return 0.0
        occupied = sum(1 for t in self._tables if t.status == TableStatus.OCCUPIED)
        return occupied / len(self._tables)

    def _tick_dining_timers(self) -> int:
        freed = 0
        to_free = []
        for tid, remaining in list(self._dining_timers.items()):
            self._dining_timers[tid] = remaining - 1
            if self._dining_timers[tid] <= 0:
                to_free.append(tid)

        for tid in to_free:
            del self._dining_timers[tid]
            table = self._get_table(tid)
            if table:
                table.status = TableStatus.AVAILABLE
                table.party_size = 0
                table.time_seated = 0
                freed += 1
        return freed

    def _spawn_customers(self) -> int:
        arrival_rate = self._cfg["arrival_rate"]
        max_patience = self._cfg["max_patience"]
        arrived = 0
        # Poisson-like: check if a new group arrives
        if self._rng.random() < arrival_rate:
            party_size = self._rng.choices(
                [1, 2, 3, 4, 5, 6, 7, 8],
                weights=[5, 25, 20, 20, 10, 8, 7, 5],
            )[0]
            patience = self._rng.randint(max(1, max_patience - 2), max_patience + 2)
            revenue = party_size * BASE_REVENUE_PER_PERSON * self._rng.uniform(0.8, 1.3)
            self._customer_counter += 1
            self._queue.append(Customer(
                id=self._customer_counter,
                party_size=party_size,
                patience_remaining=patience,
                arrival_step=self._time_step,
                revenue_value=round(revenue, 2),
            ))
            arrived += 1
        return arrived

    def _process_walkouts(self) -> int:
        """Decrement patience; remove customers who run out."""
        walked = 0
        remaining = []
        for customer in self._queue:
            customer.patience_remaining -= 1
            if customer.patience_remaining <= 0:
                walked += 1
                self._total_walkouts += 1
            else:
                remaining.append(customer)
        self._queue = remaining
        return walked

    def _execute_action(
        self,
        action: str,
        table_id: Optional[int],
        combine_with: Optional[int],
    ) -> Tuple[float, Dict[str, Any]]:
        """Execute the agent's chosen action and return (reward, info)."""
        info: Dict[str, Any] = {}

        if action == Action.ASSIGN_TABLE:
            return self._action_assign(table_id, info)

        elif action == Action.REJECT_CUSTOMER:
            return self._action_reject(info)

        elif action == Action.DELAY_SEATING:
            info["note"] = "delay"
            return -0.5, info

        elif action == Action.COMBINE_TABLES:
            return self._action_combine(table_id, combine_with, info)

        return 0.0, info

    def _action_assign(self, table_id: Optional[int], info: Dict) -> Tuple[float, Dict]:
        if not self._queue:
            info["note"] = "queue_empty"
            return -0.2, info

        customer = self._queue[0]

        # Auto-select best-fit table if none specified
        target = None
        if table_id is not None:
            target = self._get_table(table_id)
        else:
            target = self._best_fit_table(customer.party_size)

        if target is None or target.status != TableStatus.AVAILABLE:
            info["note"] = "no_suitable_table"
            return -1.0, info

        if target.capacity < customer.party_size:
            info["note"] = "table_too_small"
            return -1.0, info

        # Seat the customer
        self._queue.pop(0)
        target.status = TableStatus.OCCUPIED
        target.party_size = customer.party_size
        target.time_seated = 0
        target.revenue_earned += customer.revenue_value

        # Dining timer
        dining_time = self._rng.randint(MIN_DINING_STEPS, MAX_DINING_STEPS)
        self._dining_timers[target.id] = dining_time

        self._total_seated += 1
        self._total_revenue += customer.revenue_value

        # Reward: base + efficiency bonus for tight fit
        fit_efficiency = customer.party_size / target.capacity
        reward = 10.0 + (fit_efficiency * 5.0) + (customer.revenue_value / 50.0)

        # Speed bonus: seat quickly (full patience = max bonus)
        wait_steps = self._time_step - customer.arrival_step
        speed_bonus = max(0, 3.0 - wait_steps * 0.5)
        reward += speed_bonus

        info["seated_customer_id"] = customer.id
        info["table_id"] = target.id
        info["revenue"] = customer.revenue_value
        info["fit_efficiency"] = round(fit_efficiency, 3)
        return round(reward, 4), info

    def _action_reject(self, info: Dict) -> Tuple[float, Dict]:
        if not self._queue:
            info["note"] = "queue_empty_reject"
            return -0.2, info
        removed = self._queue.pop(0)
        self._total_rejected += 1
        info["rejected_customer_id"] = removed.id
        return -5.0, info

    def _action_combine(
        self,
        table_id: Optional[int],
        combine_with: Optional[int],
        info: Dict,
    ) -> Tuple[float, Dict]:
        if table_id is None or combine_with is None:
            info["note"] = "combine_needs_two_ids"
            return -1.0, info

        t1 = self._get_table(table_id)
        t2 = self._get_table(combine_with)

        if t1 is None or t2 is None:
            info["note"] = "combine_invalid_ids"
            return -1.0, info

        if t1.status != TableStatus.AVAILABLE or t2.status != TableStatus.AVAILABLE:
            info["note"] = "combine_tables_not_free"
            return -1.5, info

        if not self._queue:
            info["note"] = "combine_queue_empty"
            return -0.5, info

        customer = self._queue[0]
        combined_capacity = t1.capacity + t2.capacity

        if combined_capacity < customer.party_size:
            info["note"] = "combined_still_too_small"
            return -1.0, info

        # Combine: mark both tables, seat in t1
        self._queue.pop(0)
        t1.status = TableStatus.OCCUPIED
        t1.party_size = customer.party_size
        t1.combined_with = t2.id
        t1.revenue_earned += customer.revenue_value

        t2.status = TableStatus.COMBINED
        t2.combined_with = t1.id

        dining_time = self._rng.randint(MIN_DINING_STEPS, MAX_DINING_STEPS)
        self._dining_timers[t1.id] = dining_time

        self._total_seated += 1
        self._total_revenue += customer.revenue_value

        reward = 8.0 + (customer.revenue_value / 50.0)  # slightly lower than perfect fit
        info["combined_tables"] = [t1.id, t2.id]
        info["combined_capacity"] = combined_capacity
        return round(reward, 4), info

    def _best_fit_table(self, party_size: int) -> Optional[Table]:
        """Select smallest available table that fits the party (best-fit heuristic)."""
        candidates = [
            t for t in self._tables
            if t.status == TableStatus.AVAILABLE and t.capacity >= party_size
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t.capacity)

    def _get_table(self, table_id: int) -> Optional[Table]:
        for t in self._tables:
            if t.id == table_id:
                return t
        return None
