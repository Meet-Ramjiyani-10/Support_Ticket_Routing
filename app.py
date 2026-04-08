"""
FastAPI app — exposes the OpenEnv HTTP interface for the Support Routing Environment.
Endpoints: POST /reset  POST /step  GET /state  GET /tasks  GET /health
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from env import SupportRoutingEnv, Action, Observation, Reward

app = FastAPI(
    title="Support Ticket Routing — OpenEnv",
    description="An OpenEnv environment for training agents to route customer support tickets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instances per task
_envs: dict[str, SupportRoutingEnv] = {}


def _get_env(task: str) -> SupportRoutingEnv:
    if task not in SupportRoutingEnv.TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task}'. Valid: {SupportRoutingEnv.TASKS}")
    if task not in _envs:
        _envs[task] = SupportRoutingEnv(task=task)
    return _envs[task]


# ── Request / Response schemas ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "task_easy"
    seed: int = 42

class StepRequest(BaseModel):
    task: str = "task_easy"
    action: Action

class StepResponse(BaseModel):
    observation: Optional[Observation]
    reward: Reward
    done: bool
    info: dict

class GradeResponse(BaseModel):
    task: str
    score: float
    history_length: int


# ── Endpoints ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "support-routing"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "task_easy",   "description": "Route 3 clearly-signaled tickets",          "difficulty": "easy"},
            {"id": "task_medium", "description": "Route 3 ambiguous tickets with tier signals","difficulty": "medium"},
            {"id": "task_hard",   "description": "Route 3 edge-case tickets (legal/security)", "difficulty": "hard"},
        ]
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    env = SupportRoutingEnv(task=req.task, seed=req.seed)
    _envs[req.task] = env
    obs = env.reset()
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.task)
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state(task: str = "task_easy"):
    env = _get_env(task)
    return env.state()


@app.get("/grade", response_model=GradeResponse)
def grade(task: str = "task_easy"):
    env = _get_env(task)
    score = env.grade()
    return GradeResponse(task=task, score=score, history_length=len(env._history))
