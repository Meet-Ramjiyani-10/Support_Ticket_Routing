"""
Inference Script — Support Ticket Routing Environment
======================================================
MANDATORY setup:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.

Run:
  python inference.py

Expected output: scores for task_easy, task_medium, task_hard
"""

import os
import json
import requests
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert customer support manager.
You will be given a support ticket and must route it correctly.

Respond ONLY with a JSON object in this exact format:
{
  "queue": "<one of: billing, technical, general, sales, abuse>",
  "priority": "<one of: low, medium, high, critical>",
  "requires_human": <true or false>,
  "notes": "<brief reason for your routing decision>"
}

Routing rules:
- billing: payment issues, invoices, refunds, cancellations
- technical: bugs, crashes, API issues, login problems, security incidents
- general: how-to questions, feature requests, documentation
- sales: upgrade inquiries, enterprise interest, pricing questions
- abuse: GDPR violations, scraping, ToS violations, legal threats

Priority rules:
- critical: enterprise customers with blockers, legal threats, security breaches, mass cancellations
- high: paying customers blocked from core features, billing errors, potential churn
- medium: degraded experience, workarounds exist
- low: feature requests, general questions

requires_human: true if the issue needs a human agent (financial impact, legal, security, churn risk)
"""

def route_ticket(obs: dict) -> dict:
    """Ask the LLM to route a single ticket."""
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
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    action = json.loads(raw)
    return action


def run_task(task_id: str) -> float:
    """Run one full episode of a task and return the grade score."""
    print(f"\n{'='*50}")
    print(f"Task: {task_id}")
    print(f"{'='*50}")

    # Reset
    reset_resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task_id, "seed": 42})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    step_num = 0
    done = False

    while not done:
        step_num += 1
        print(f"\n--- Ticket {step_num} ---")
        print(f"  Subject : {obs['subject']}")
        print(f"  Tier    : {obs['customer_tier']}")

        action = route_ticket(obs)
        print(f"  Routed  : queue={action['queue']} | priority={action['priority']} | human={action['requires_human']}")

        step_resp = requests.post(f"{ENV_BASE_URL}/step", json={
            "task": task_id,
            "action": action,
        })
        step_resp.raise_for_status()
        result = step_resp.json()

        reward_val = result["reward"]["value"]
        feedback   = result["reward"]["feedback"]
        print(f"  Reward  : {reward_val:.2f} — {feedback}")

        done = result["done"]
        if not done:
            obs = result["observation"]

    # Grade
    grade_resp = requests.get(f"{ENV_BASE_URL}/grade", params={"task": task_id})
    grade_resp.raise_for_status()
    score = grade_resp.json()["score"]
    print(f"\n✅ Final score for {task_id}: {score:.4f}")
    return score


def main():
    print("Support Ticket Routing — Baseline Inference")
    print(f"Model   : {MODEL_NAME}")
    print(f"Env URL : {ENV_BASE_URL}")

    tasks = ["task_easy", "task_medium", "task_hard"]
    scores = {}

    for task in tasks:
        try:
            scores[task] = run_task(task)
        except Exception as e:
            print(f"ERROR on {task}: {e}")
            scores[task] = 0.0

    print("\n" + "="*50)
    print("BASELINE SCORES SUMMARY")
    print("="*50)
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<15} {score:.4f}  {bar}")
    overall = sum(scores.values()) / len(scores)
    print(f"\n  Overall avg: {overall:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
