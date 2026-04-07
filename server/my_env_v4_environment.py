"""
server/my_env_v4_environment.py
OpenEnv environment registration shim.

This module satisfies the OpenEnv framework requirement for a discoverable
environment class. It wraps RestaurantEnv and exposes a standard interface.
"""
from __future__ import annotations

from server.environment import RestaurantEnv, TASK_CONFIGS

# Public export expected by OpenEnv loader
Environment = RestaurantEnv
TASK_IDS    = list(TASK_CONFIGS.keys())

__all__ = ["Environment", "TASK_IDS", "TASK_CONFIGS"]
