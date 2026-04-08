"""
Customer Support Ticket Routing Environment
OpenEnv-compliant environment for training/evaluating AI agents on support ticket triage.
"""

import random
import json
from typing import Optional, Any
from pydantic import BaseModel, Field


# ── Typed Models ────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: str          # "free" | "pro" | "enterprise"
    previous_contacts: int      # how many times this customer contacted before
    created_at: str             # ISO timestamp string
    available_queues: list[str]

class Action(BaseModel):
    queue: str                  # which queue to route to
    priority: str               # "low" | "medium" | "high" | "critical"
    requires_human: bool        # escalate to human agent?
    notes: Optional[str] = ""   # optional agent notes

class Reward(BaseModel):
    value: float                # 0.0 – 1.0
    breakdown: dict[str, float]
    feedback: str


# ── Ticket Dataset ───────────────────────────────────────────────────────────────

QUEUES = ["billing", "technical", "general", "sales", "abuse"]

TICKET_TEMPLATES = [
    # Easy tickets — clear signals
    {
        "subject": "Invoice #4521 is incorrect",
        "body": "Hi, I was charged $99 but my plan is $49/month. Please fix this immediately.",
        "customer_tier": "pro",
        "correct_queue": "billing",
        "correct_priority": "high",
        "requires_human": True,
        "difficulty": "easy",
    },
    {
        "subject": "Cannot login to my account",
        "body": "I've been trying to login for 2 hours. Password reset doesn't work either.",
        "customer_tier": "free",
        "correct_queue": "technical",
        "correct_priority": "medium",
        "requires_human": False,
        "difficulty": "easy",
    },
    {
        "subject": "How do I export my data?",
        "body": "I'd like to download all my data as a CSV. Where can I find this option?",
        "customer_tier": "free",
        "correct_queue": "general",
        "correct_priority": "low",
        "requires_human": False,
        "difficulty": "easy",
    },
    {
        "subject": "Interested in upgrading our plan",
        "body": "We have 50 employees and want to discuss enterprise pricing. Can someone call us?",
        "customer_tier": "pro",
        "correct_queue": "sales",
        "correct_priority": "high",
        "requires_human": True,
        "difficulty": "easy",
    },
    # Medium tickets — some ambiguity
    {
        "subject": "App keeps crashing — losing business",
        "body": "Your app crashes every time I try to generate a report. We have a client demo tomorrow and this is unacceptable. I pay for enterprise.",
        "customer_tier": "enterprise",
        "correct_queue": "technical",
        "correct_priority": "critical",
        "requires_human": True,
        "difficulty": "medium",
    },
    {
        "subject": "Refund request",
        "body": "I cancelled my subscription 3 days ago and was still charged. I want a full refund.",
        "customer_tier": "pro",
        "correct_queue": "billing",
        "correct_priority": "high",
        "requires_human": True,
        "difficulty": "medium",
    },
    {
        "subject": "Strange activity on my account",
        "body": "I noticed logins from locations I don't recognize. Someone may have accessed my data.",
        "customer_tier": "enterprise",
        "correct_queue": "technical",
        "correct_priority": "critical",
        "requires_human": True,
        "difficulty": "medium",
    },
    {
        "subject": "Feature request: dark mode",
        "body": "Would love a dark mode option. My eyes get tired. Many users in the community forum agree.",
        "customer_tier": "free",
        "correct_queue": "general",
        "correct_priority": "low",
        "requires_human": False,
        "difficulty": "medium",
    },
    # Hard tickets — tricky routing
    {
        "subject": "You are violating GDPR",
        "body": "I submitted a data deletion request 35 days ago (legal requirement: 30 days). My data is still visible. I will escalate to the data protection authority.",
        "customer_tier": "free",
        "correct_queue": "abuse",
        "correct_priority": "critical",
        "requires_human": True,
        "difficulty": "hard",
    },
    {
        "subject": "API rate limits too low",
        "body": "We're hitting rate limits constantly. Our integration breaks. We need higher limits or we'll move to a competitor.",
        "customer_tier": "enterprise",
        "correct_queue": "technical",
        "correct_priority": "high",
        "requires_human": True,
        "difficulty": "hard",
    },
    {
        "subject": "Bulk account cancellation",
        "body": "Please cancel all 47 accounts under our organization. We are terminating our contract immediately due to the recent security incident.",
        "customer_tier": "enterprise",
        "correct_queue": "billing",
        "correct_priority": "critical",
        "requires_human": True,
        "difficulty": "hard",
    },
    {
        "subject": "Competitor is scraping our data",
        "body": "I believe a competitor is using your platform to scrape my company's public listings in bulk. I have evidence.",
        "customer_tier": "pro",
        "correct_queue": "abuse",
        "correct_priority": "high",
        "requires_human": True,
        "difficulty": "hard",
    },
]

PRIORITY_ORDER = ["low", "medium", "high", "critical"]


# ── Environment ──────────────────────────────────────────────────────────────────

class SupportRoutingEnv:
    """
    OpenEnv-compliant Customer Support Ticket Routing Environment.

    Tasks:
      - task_easy:   Route 3 clearly-signaled tickets correctly
      - task_medium: Route 3 ambiguous tickets with tier awareness
      - task_hard:   Route 3 edge-case tickets (legal, security, churn)
    """

    TASKS = ["task_easy", "task_medium", "task_hard"]

    def __init__(self, task: str = "task_easy", seed: int = 42):
        assert task in self.TASKS, f"Unknown task: {task}"
        self.task = task
        self.seed = seed
        self._rng = random.Random(seed)
        self._tickets: list[dict] = []
        self._current_idx: int = 0
        self._history: list[dict] = []
        self._done: bool = False

    def _difficulty_for_task(self) -> str:
        return self.task.replace("task_", "")  # easy / medium / hard

    def _build_ticket_pool(self) -> list[dict]:
        difficulty = self._difficulty_for_task()
        pool = [t for t in TICKET_TEMPLATES if t["difficulty"] == difficulty]
        self._rng.shuffle(pool)
        return pool[:3]  # exactly 3 tickets per episode

    def reset(self) -> Observation:
        self._rng = random.Random(self.seed)
        self._tickets = self._build_ticket_pool()
        self._current_idx = 0
        self._history = []
        self._done = False
        return self._make_observation()

    def _make_observation(self) -> Observation:
        t = self._tickets[self._current_idx]
        return Observation(
            ticket_id=f"TKT-{1000 + self._current_idx}",
            subject=t["subject"],
            body=t["body"],
            customer_tier=t["customer_tier"],
            previous_contacts=self._rng.randint(0, 5),
            created_at="2025-04-01T10:00:00Z",
            available_queues=QUEUES,
        )

    def step(self, action: Action) -> tuple[Optional[Observation], Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        ticket = self._tickets[self._current_idx]
        reward = self._compute_reward(action, ticket)

        self._history.append({
            "ticket": ticket,
            "action": action.model_dump(),
            "reward": reward.model_dump(),
        })

        self._current_idx += 1
        if self._current_idx >= len(self._tickets):
            self._done = True
            return None, reward, True, {"history": self._history}

        next_obs = self._make_observation()
        return next_obs, reward, False, {}

    def state(self) -> dict:
        return {
            "task": self.task,
            "current_ticket_index": self._current_idx,
            "total_tickets": len(self._tickets),
            "done": self._done,
            "history": self._history,
        }

    def _compute_reward(self, action: Action, ticket: dict) -> Reward:
        breakdown = {}

        # 1. Queue correctness (40%)
        if action.queue == ticket["correct_queue"]:
            breakdown["queue"] = 0.40
        else:
            breakdown["queue"] = 0.0

        # 2. Priority correctness (30%) — partial credit for adjacent priority
        correct_p = ticket["correct_priority"]
        agent_p = action.priority
        p_idx_correct = PRIORITY_ORDER.index(correct_p)
        p_idx_agent = PRIORITY_ORDER.index(agent_p) if agent_p in PRIORITY_ORDER else -1
        diff = abs(p_idx_correct - p_idx_agent)
        if diff == 0:
            breakdown["priority"] = 0.30
        elif diff == 1:
            breakdown["priority"] = 0.15  # partial credit
        else:
            breakdown["priority"] = 0.0

        # 3. Human escalation (20%)
        if action.requires_human == ticket["requires_human"]:
            breakdown["escalation"] = 0.20
        else:
            # Penalize harder for missing a required escalation
            breakdown["escalation"] = 0.0 if ticket["requires_human"] else 0.10

        # 4. Notes quality bonus (10%) — reward non-empty notes
        if action.notes and len(action.notes.strip()) > 10:
            breakdown["notes"] = 0.10
        else:
            breakdown["notes"] = 0.0

        total = sum(breakdown.values())
        total = round(min(max(total, 0.0), 1.0), 4)

        # Build human-readable feedback
        issues = []
        if breakdown["queue"] == 0:
            issues.append(f"wrong queue (expected '{ticket['correct_queue']}')")
        if breakdown["priority"] < 0.30:
            issues.append(f"wrong priority (expected '{ticket['correct_priority']}')")
        if breakdown["escalation"] == 0:
            issues.append(f"wrong escalation (expected {ticket['requires_human']})")
        feedback = "Perfect routing!" if not issues else "Issues: " + "; ".join(issues)

        return Reward(value=total, breakdown=breakdown, feedback=feedback)

    # ── Graders (called by OpenEnv evaluate) ────────────────────────────────────

    def grade(self) -> float:
        """Return mean reward across the episode. Call after episode ends."""
        if not self._history:
            return 0.0
        return round(sum(h["reward"]["value"] for h in self._history) / len(self._history), 4)
