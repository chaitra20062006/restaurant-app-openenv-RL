"""
server/app.py - FastAPI + Gradio Unified Server
"""
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

# Import your existing models and environment
from models import ResetRequest, ResetResponse, StepRequest, StepResponse, StateResponse, HealthResponse
from server.environment import RestaurantEnv, TASK_CONFIGS
from gradio_ui import create_demo # Import the function we just created

app = FastAPI(title="Restaurant OpenEnv API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENV = RestaurantEnv()

@app.get("/health")
async def health(): return {"status": "healthy"}

@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest):
    obs = ENV.reset(task=req.task, seed=req.seed)
    return ResetResponse(task=req.task, observation=obs)

@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    obs, reward, done, info = ENV.step(action=req.action, table_id=req.table_id, combine_with=req.combine_with)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

# MOUNT GRADIO: Dashboard at '/', API at '/reset', '/step', '/docs'
app = gr.mount_gradio_app(app, create_demo(), path="/")

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()