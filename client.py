"""
client.py
Simple HTTP client wrapper for the Restaurant Table Management OpenEnv API.
Useful for quick manual testing or scripting without running inference.py.

Usage:
  python client.py reset --task easy
  python client.py step --action assign_table
  python client.py state
  python client.py health
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _pretty(data: dict) -> str:
    return json.dumps(data, indent=2)


def cmd_health(args):
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    r.raise_for_status()
    print(_pretty(r.json()))


def cmd_reset(args):
    payload = {"task": args.task}
    if args.seed:
        payload["seed"] = int(args.seed)
    r = requests.post(f"{BASE_URL}/reset", json=payload, timeout=15)
    r.raise_for_status()
    print(_pretty(r.json()))


def cmd_step(args):
    payload = {"action": args.action}
    if args.table_id:
        payload["table_id"] = int(args.table_id)
    if args.combine_with:
        payload["combine_with"] = int(args.combine_with)
    r = requests.post(f"{BASE_URL}/step", json=payload, timeout=15)
    r.raise_for_status()
    print(_pretty(r.json()))


def cmd_state(args):
    r = requests.get(f"{BASE_URL}/state", timeout=10)
    r.raise_for_status()
    print(_pretty(r.json()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restaurant Env CLI Client")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health")

    p_reset = sub.add_parser("reset")
    p_reset.add_argument("--task",   default="easy", choices=["easy", "medium", "hard"])
    p_reset.add_argument("--seed",   default=None)

    p_step = sub.add_parser("step")
    p_step.add_argument("--action", required=True,
                        choices=["assign_table", "reject_customer",
                                 "delay_seating", "combine_tables"])
    p_step.add_argument("--table-id",    default=None)
    p_step.add_argument("--combine-with", default=None)

    sub.add_parser("state")

    args = parser.parse_args()
    {"health": cmd_health, "reset": cmd_reset,
     "step": cmd_step, "state": cmd_state}[args.command](args)
