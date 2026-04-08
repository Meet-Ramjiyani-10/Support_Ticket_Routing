# 🎫 Customer Support Ticket Routing — OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://openenv.dev)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)

An **OpenEnv-compliant** environment that trains and evaluates AI agents on the real-world task of **customer support ticket routing** — deciding where to send a ticket, how urgently, and whether it needs a human.

---

## 🌍 Why This Environment?

Every company with customers deals with ticket routing. Misrouting costs time, money, and customer trust. Getting it right requires reading natural language, understanding context (customer tier, urgency signals, legal risk), and applying business rules — a perfect challenge for LLM agents.

---

## 🗂️ Environment Overview

| Property | Value |
|---|---|
| Task type | NLP / Decision making |
| Action space | Discrete (queue + priority + escalation) |
| Observation space | Structured ticket data |
| Episode length | 3 tickets |
| Reward | Continuous 0.0–1.0 |

---

## 🎯 Tasks

| Task ID | Difficulty | Description |
|---|---|---|
| `task_easy` | Easy | 3 tickets with obvious routing signals |
| `task_medium` | Medium | 3 tickets requiring tier-awareness & context |
| `task_hard` | Hard | 3 edge-case tickets: legal, security, churn |

---

## 👁️ Observation Space

```json
{
  "ticket_id": "TKT-1000",
  "subject": "Invoice is incorrect",
  "body": "I was charged $99 but my plan is $49...",
  "customer_tier": "pro",
  "previous_contacts": 2,
  "created_at": "2025-04-01T10:00:00Z",
  "available_queues": ["billing", "technical", "general", "sales", "abuse"]
}
```

## ⚡ Action Space

```json
{
  "queue": "billing",
  "priority": "high",
  "requires_human": true,
  "notes": "Billing error for paying customer — needs immediate review"
}
```

**Queues:** `billing` | `technical` | `general` | `sales` | `abuse`  
**Priorities:** `low` | `medium` | `high` | `critical`

---

## 🏆 Reward Function

| Component | Weight | Description |
|---|---|---|
| Queue correctness | 40% | Routed to the right team |
| Priority correctness | 30% | Correct urgency (partial credit for ±1 level) |
| Escalation correctness | 20% | Correct human escalation decision |
| Notes quality | 10% | Non-empty, meaningful notes provided |

Partial credit is awarded for priority within 1 level of correct — encouraging agents to reason about urgency rather than guess.

---

## 🚀 Setup & Usage

### Local

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t support-routing-env .
docker run -p 7860:7860 support-routing-env
```

### API

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "task_easy", "seed": 42}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task": "task_easy",
    "action": {
      "queue": "billing",
      "priority": "high",
      "requires_human": true,
      "notes": "Billing discrepancy for pro customer"
    }
  }'

# Grade
curl http://localhost:7860/grade?task=task_easy
```

---

## 🤖 Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

### Baseline Scores (Llama-3.3-70B)

| Task | Score |
|---|---|
| task_easy | ~0.85 |
| task_medium | ~0.72 |
| task_hard | ~0.58 |

---

## 📁 Project Structure

```
support-routing-env/
├── env.py           # Core environment logic
├── app.py           # FastAPI HTTP interface
├── inference.py     # Baseline inference script
├── openenv.yaml     # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```
