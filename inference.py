"""
inference.py
Fixed version – Uses Hugging Face Inference API (NO OpenAI client)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional, List

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")

MODEL_NAME: str = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")

HF_TOKEN: str = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set. Use: setx HF_TOKEN your_token")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def print_err(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)

VALID_ACTIONS = ["assign_table", "reject_customer", "delay_seating", "combine_tables"]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert restaurant manager AI.
Decide the best action for the current state.
Respond ONLY with a valid JSON object:
{
  "action": "assign_table|reject_customer|delay_seating|combine_tables",
  "reasoning": "one sentence explanation"
}"""

def _build_user_message(obs: Dict[str, Any]) -> str:
    tables = obs.get("tables", [])
    queue = obs.get("waiting_queue", [])
    avail = [t for t in tables if t.get("status") == "available"]

    return (
        f"Step: {obs.get('time_step', 0)} | Occupancy: {obs.get('occupancy_rate', 0):.0%}\n"
        f"Available: {[t['id'] for t in avail]}\n"
        f"Queue: {[c['party_size'] for c in queue[:3]]}\n"
    )

# ---------------------------------------------------------------------------
# LLM Decision (FIXED)
# ---------------------------------------------------------------------------
def llm_decide(obs: Dict[str, Any]) -> Dict[str, Any]:
    from huggingface_hub import InferenceClient
    
    user_msg = _build_user_message(obs)

    # Use the official Python client. It automatically manages router URLs!
    client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": f"{user_msg}\n\nRespond ONLY with a valid JSON object."}
        ]
        
        try:
            # Modern chat models (Llama 3, Qwen, Gamma) support conversational task natively
            response = client.chat_completion(
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            text = response.choices[0].message.content
        except Exception as chat_err:
            # If the provider says "not supported for conversational", fallback to raw text-generation!
            prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}\n\nRespond ONLY in JSON format."
            text = client.text_generation(
                prompt,
                max_new_tokens=150,
                temperature=0.7,
                return_full_text=False
            )

        # Extract JSON safely
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]

        decision = json.loads(text)

        if decision.get("action") not in VALID_ACTIONS:
            raise ValueError("Invalid action")

        return decision

    except Exception as e:
        print_err(f"LLM failed ({e}). Using heuristic.")
        return _heuristic_fallback(obs)

# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------
def _heuristic_fallback(obs: Dict[str, Any]) -> Dict[str, Any]:
    tables = obs.get("tables", [])
    queue = obs.get("waiting_queue", [])
    avail = [t for t in tables if t.get("status") == "available"]

    if queue and avail:
        party = queue[0]["party_size"]
        if any(t["capacity"] >= party for t in avail):
            return {"action": "assign_table", "reasoning": "Seating available party"}
    return {"action": "delay_seating", "reasoning": "Waiting for better fit"}

# ---------------------------------------------------------------------------
# Episode Runner
# ---------------------------------------------------------------------------
def run_episode(task: str = "easy", seed: Optional[int] = None) -> Dict[str, Any]:
    reset_payload = {"task": task, "seed": seed}
    resp = requests.post(f"{API_BASE_URL}/reset", json=reset_payload, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observation"]

    log_start(task=task, env="restaurant-table-management", model=MODEL_NAME)

    total_reward, step, done, rewards = 0.0, 0, False, []

    while not done:
        decision = llm_decide(obs)
        action = decision["action"]

        payload = {"action": action}

        if action == "combine_tables":
            tables = obs.get("tables", [])
            smalls = [t for t in tables if t.get("status") == "available" and t["capacity"] <= 4]
            if len(smalls) >= 2:
                payload["table_id"] = smalls[0]["id"]
                payload["combine_with"] = smalls[1]["id"]
            else:
                payload["action"] = "delay_seating"

        step_resp = requests.post(f"{API_BASE_URL}/step", json=payload, timeout=30)
        step_resp.raise_for_status()
        data = step_resp.json()

        obs, reward, done = data["observation"], data["reward"], data["done"]

        total_reward += reward
        step += 1
        rewards.append(reward)

        log_step(step, action, reward, done, None)

        if step > 200:
            break

    return {
        "task": task,
        "steps": step,
        "total_reward": total_reward,
        "rewards": rewards,
        "final_obs": obs,
    }

# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def grade_after_episode(obs: Dict[str, Any], task: str) -> float:
    seated = obs.get("total_seated", 0)
    total = seated + obs.get("total_rejected", 0) + obs.get("total_walkouts", 0)
    return round(seated / total if total > 0 else 0, 4)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    res = run_episode(task=args.task, seed=args.seed)
    score = grade_after_episode(res["final_obs"], args.task)

    log_end(success=(score >= 0.5), steps=res["steps"], score=score, rewards=res["rewards"])