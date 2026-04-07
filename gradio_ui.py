"""
gradio_ui.py - Unified version for Restaurant Table Management.
"""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Tuple
import gradio as gr
import requests

# Use 127.0.0.1 to avoid Windows localhost resolution delays
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")

# --- API Helpers ---
def _post(endpoint: str, payload: Dict) -> Dict:
    r = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def api_reset(task: str, seed: int | None = None) -> Dict:
    payload: Dict[str, Any] = {"task": task}
    if seed: payload["seed"] = int(seed)
    return _post("/reset", payload)

def api_step(action: str, table_id: int | None = None, combine_with: int | None = None) -> Dict:
    payload: Dict[str, Any] = {"action": action}
    if table_id is not None: payload["table_id"] = table_id
    if combine_with is not None: payload["combine_with"] = combine_with
    return _post("/step", payload)

def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except: return False

# --- Rendering Helpers ---
STATUS_EMOJI = {"available": "🟢", "occupied": "🔴", "combined": "🟡", "reserved": "🔵"}

def render_tables_html(tables: List[Dict]) -> str:
    if not tables: return "<p style='color:#888'>Reset environment to view tables.</p>"
    cards = []
    for t in tables:
        status = t.get("status", "available")
        emoji = STATUS_EMOJI.get(status, "⬜")
        border = {"available": "#2ecc71", "occupied": "#e74c3c", "combined": "#f39c12"}.get(status, "#555")
        cards.append(f"""
        <div style="background:#1a1a1a; border:2px solid {border}; border-radius:10px; padding:10px; min-width:100px; text-align:center; color:white;">
            <div style="font-size:20px">{emoji}</div>
            <div style="font-weight:bold">T{t['id']} ({t['capacity']} seats)</div>
            <div style="font-size:12px; color:#aaa">{status}</div>
        </div>
        """)
    return f"<div style='display:flex; flex-wrap:wrap; gap:10px; background:#0d1117; padding:15px; border-radius:10px;'>{''.join(cards)}</div>"

def render_queue_html(queue: List[Dict]) -> str:
    if not queue: return "<p style='color:#2ecc71'>✓ Queue empty</p>"
    rows = "".join([f"<li>Group of {c['party_size']} (Patience: {c['patience_remaining']})</li>" for c in queue])
    return f"<ul style='color:white; font-family:monospace;'>{rows}</ul>"

# --- Gradio UI Layout ---
with gr.Blocks(title="Restaurant Table Manager") as demo:
    env_state = gr.State({})
    done_flag = gr.State(False)
    
    gr.Markdown("# 🍽️ Restaurant Table Manager Dashboard")
    
    with gr.Row():
        with gr.Column(scale=1):
            task_dd = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Task")
            reset_btn = gr.Button("RESET", variant="primary")
            auto_btn = gr.Button("Auto-Step")
        with gr.Column(scale=2):
            tables_display = gr.HTML()
            queue_display = gr.HTML()
            log_box = gr.Textbox(label="Logs", lines=10, interactive=False)

    def do_reset(task):
        data = api_reset(task)
        obs = data["observation"]
        return obs, render_tables_html(obs["tables"]), render_queue_html(obs["waiting_queue"]), "[START] Episode initialized."

    reset_btn.click(do_reset, inputs=[task_dd], outputs=[env_state, tables_display, queue_display, log_box])

# Export for FastAPI mounting
def create_demo():
    return demo