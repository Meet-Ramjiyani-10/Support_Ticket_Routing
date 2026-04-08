"""
Inference Script — Support Ticket Routing Environment
OpenEnv Hackathon Compliant Version
"""

import os
import json
import requests
from openai import OpenAI

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# HF Space URL
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://meetppatel-support-ticket-routing.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert customer support manager.
You will be given a support ticket and must route it correctly.

Respond ONLY with a JSON object in this exact format:
{
  "queue": "<one of: billing, technical, general, sales, abuse>",
  "priority": "<one of: low, medium, high, critical>",
  "requires_human": <true or false>,
  "notes": "<brief reason for your routing decision>"
}"""

def route_ticket(obs: dict) -> dict:
    user_msg = f"""Route this support ticket:

Subject: {obs['subject']}
Body: {obs['body']}
Customer tier: {obs['customer_tier']}
Previous contacts: {obs['previous_contacts']}
Available queues: {', '.join(obs['available_queues'])}
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)

def run_task(task_id: str):
    print(f"[START] task={task_id} env=support-ticket-routing model={MODEL_NAME}")
    
    reset_resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task_id, "seed": 42})
    reset_resp.raise_for_status()
    obs = reset_resp.json()
    
    step_num = 0
    done = False
    rewards = []
    error = None

    while not done:
        step_num += 1
        try:
            action = route_ticket(obs)
            action_str = f"queue={action['queue']},priority={action['priority']},human={action['requires_human']}"
            
            step_resp = requests.post(f"{ENV_BASE_URL}/step", json={
                "task": task_id,
                "action": action,
            })
            step_resp.raise_for_status()
            result = step_resp.json()
            
            reward_val = result["reward"]["value"]
            done = result["done"]
            rewards.append(reward_val)
            error = None
            
        except Exception as e:
            error = str(e)
            reward_val = 0.0
            done = True
            rewards.append(reward_val)
            action_str = "error"
        
        print(f"[STEP] step={step_num} action={action_str} reward={reward_val:.2f} done={str(done).lower()} error={error if error else 'null'}")
        
        if not done and 'result' in locals() and result.get("observation"):
            obs = result["observation"]
    
    success = len(rewards) > 0 and all(r > 0.5 for r in rewards) if rewards else False
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}")

def main():
    tasks = ["task_easy", "task_medium", "task_hard"]
    for task in tasks:
        try:
            run_task(task)
        except Exception as e:
            print(f"[END] success=false steps=0 rewards= error={str(e)}")

if __name__ == "__main__":
    main()
