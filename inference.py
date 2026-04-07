#!/usr/bin/env python3
"""Baseline inference script for DNS-Env (OpenEnv Hackathon).

STDOUT FORMAT (mandatory — any deviation = incorrect scoring):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from typing import Any, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["fix_single_record", "configure_mail", "debug_delegation"]
BENCHMARK = "dns_env"

# Safety limits
MAX_STEPS_PER_TASK = 25
MAX_RETRIES_HTTP = 3
HTTP_TIMEOUT = 60
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", ""),
)

# ---------------------------------------------------------------------------
# Mandatory stdout log functions
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
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert DNS administrator and zone-file debugger. You are \
interacting with a DNS zone-file debugging environment. Your goal is to \
diagnose and fix errors in DNS zone files so that all records resolve \
correctly.

## Available Commands

Respond with exactly ONE JSON object per turn. The JSON must have \
"command" and "args" keys.

1. **view_zone** -- View the zone file with indexed records.
   {"command": "view_zone", "args": {"zone": "<zone_name>"}}

2. **check_zone** -- Validate the zone and list any errors.
   {"command": "check_zone", "args": {"zone": "<zone_name>"}}

3. **edit_record** -- Modify an existing record by its index. Only \
include the fields you want to change.
   {"command": "edit_record", "args": {"zone": "<zone_name>", "index": <N>, "rdata": "<new_rdata>"}}
   You can also change "name", "rtype", or "ttl" in the same call.

4. **add_record** -- Add a new record to the zone.
   {"command": "add_record", "args": {"zone": "<zone_name>", "name": "<name>", "rtype": "<type>", "rdata": "<data>"}}

5. **delete_record** -- Remove a record by index.
   {"command": "delete_record", "args": {"zone": "<zone_name>", "index": <N>}}

6. **dig** -- Simulate a DNS query to test resolution.
   {"command": "dig", "args": {"zone": "<zone_name>", "qname": "<name>", "qtype": "<type>"}}

7. **submit** -- Submit your work for grading. Use this when you are \
confident all fixes are correct.
   {"command": "submit", "args": {}}

## DNS Debugging Tips

- **Trailing dots on FQDNs**: CNAME targets, NS targets, and MX targets \
that are fully-qualified domain names MUST end with a trailing dot \
(e.g., "example.com." not "example.com"). Without the dot, the name is \
treated as relative to the zone origin.
- **A record IPs**: Must be valid IPv4 addresses (each octet 0-255).
- **MX records**: Format is "<priority> <target_fqdn>". The target must \
end with a dot if it is an FQDN.
- **CNAME exclusivity**: A name with a CNAME record cannot have any \
other record types at the same name.
- **TXT records**: The rdata should be enclosed in double quotes.
- **SPF records**: Use TXT record type. Example: \
"v=spf1 ip4:10.0.1.0/24 -all"
- **DMARC records**: Use TXT record at _dmarc. Example: \
"v=DMARC1; p=quarantine; rua=mailto:postmaster@example.com"
- **NS delegation**: Parent zone NS records and glue records must be \
consistent with the child zone's NS records and A records.
- **SOA serial**: The child zone SOA serial should generally be >= the \
parent zone's serial.
- **Zone indices**: Records are labeled with [N] indices. Use these \
indices when editing or deleting records.

## Strategy

1. Start by viewing all available zones with view_zone.
2. Run check_zone to identify validation errors.
3. Fix each error using edit_record, add_record, or delete_record.
4. Use dig to verify your fixes resolve correctly.
5. When all issues are fixed, submit.

## Response Format

You MUST respond with a single JSON object and nothing else. \
Do not include explanations outside the JSON. Example:
{"command": "view_zone", "args": {"zone": "example.com"}}
"""

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST to the environment server with retries."""
    url = f"{ENV_URL}{endpoint}"
    for attempt in range(1, MAX_RETRIES_HTTP + 1):
        try:
            resp = requests.post(url, json=body, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES_HTTP:
                raise
            time.sleep(1.0 * attempt)
    return {}


def reset_env(task_id: str) -> dict[str, Any]:
    body: dict[str, Any] = {"session_id": "default", "options": {"task_id": task_id}}
    return _post("/reset", body)


def step_env(action: dict[str, Any]) -> dict[str, Any]:
    body = {"session_id": "default", "action": action}
    return _post("/step", body)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    obs: dict[str, Any], task_id: str, step_num: int, max_steps: int,
    history: list[dict[str, str]],
) -> str:
    parts: list[str] = []
    task_desc = obs.get("task_description", "")
    if task_desc:
        parts.append(f"## Task: {task_id}\n{task_desc}")
    zone_names = obs.get("zone_names", [])
    if zone_names:
        parts.append(f"Available zones: {', '.join(zone_names)}")
    remaining = max_steps - step_num
    parts.append(f"Step {step_num}/{max_steps} (remaining: {remaining})")
    if history:
        recent = history[-3:]
        lines = []
        for h in recent:
            lines.append(f"  Action: {h['action']}")
            preview = h["result"][:300]
            if len(h["result"]) > 300:
                preview += "..."
            lines.append(f"  Result: {preview}")
        parts.append("## Recent History\n" + "\n".join(lines))
    output = obs.get("output", "")
    if output:
        parts.append(f"## Current Output\n{output}")
    if remaining <= 3:
        parts.append(
            'WARNING: Running low on steps. Submit now: {"command": "submit", "args": {}}'
        )
    parts.append('Respond with a single JSON object: {"command": "...", "args": {...}}')
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def parse_llm_response(text: str | None) -> dict[str, Any]:
    default_action: dict[str, Any] = {"command": "view_zone", "args": {}}
    if not text:
        return default_action
    text = text.strip()
    action = _try_parse_json(text)
    if action is not None:
        return action
    match = _JSON_BLOCK_RE.search(text)
    if match:
        action = _try_parse_json(match.group(1).strip())
        if action is not None:
            return action
    match = _JSON_OBJECT_RE.search(text)
    if match:
        action = _try_parse_json(match.group(0))
        if action is not None:
            return action
    return default_action


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "command" in data:
            if "args" not in data or not isinstance(data.get("args"), dict):
                data["args"] = data.get("args", {}) or {}
            return {"command": str(data["command"]), "args": data["args"]}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def call_llm(messages: list[dict[str, str]], temperature: float = 0.0) -> str:
    for attempt in range(1, 3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=temperature,
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(2.0)
    return ""


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


def run_task(task_id: str) -> float:
    """Run a single task. Returns score in [0, 1]."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = reset_env(task_id)
    except Exception as exc:
        print(f"[ERROR] Failed to reset: {exc}", file=sys.stderr)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    max_steps = MAX_STEPS_PER_TASK
    history: list[dict[str, str]] = []
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while not obs.get("done", False):
            steps_taken += 1

            prompt = build_prompt(obs, task_id, steps_taken, max_steps, history)
            messages.append({"role": "user", "content": prompt})

            try:
                llm_text = call_llm(messages)
            except Exception:
                action = {"command": "submit", "args": {}}
                llm_text = json.dumps(action)

            messages.append({"role": "assistant", "content": llm_text})
            action = parse_llm_response(llm_text)
            history.append({"action": json.dumps(action), "result": ""})

            try:
                obs = step_env(action)
            except Exception:
                try:
                    obs = step_env({"command": "submit", "args": {}})
                except Exception:
                    obs = {"done": True, "reward": 0.0}
                break

            if history:
                history[-1]["result"] = obs.get("output", "")[:500]

            reward = obs.get("reward")
            reward_val = float(reward) if reward is not None else 0.0
            done = obs.get("done", False)
            error = None

            rewards.append(reward_val)

            action_str = json.dumps(action)
            log_step(step=steps_taken, action=action_str, reward=reward_val, done=done, error=error)

            if steps_taken >= max_steps and not obs.get("done", False):
                try:
                    obs = step_env({"command": "submit", "args": {}})
                    reward = obs.get("reward")
                    reward_val = float(reward) if reward is not None else 0.0
                    rewards.append(reward_val)
                    steps_taken += 1
                    log_step(
                        step=steps_taken,
                        action='{"command":"submit","args":{}}',
                        reward=reward_val,
                        done=obs.get("done", False),
                        error=None,
                    )
                except Exception:
                    obs = {"done": True, "reward": 0.0}
                break

            if len(messages) > 41:
                messages = [messages[0]] + messages[-40:]

        score = obs.get("reward", 0.0)
        if score is None:
            score = 0.0
        score = float(score)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    try:
        resp = requests.get(f"{ENV_URL}/health", timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[FATAL] Cannot reach environment at {ENV_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    scores: dict[str, float] = {}
    for task_id in TASKS:
        try:
            score = run_task(task_id)
            scores[task_id] = score
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            scores[task_id] = 0.0
            log_end(success=False, steps=0, score=0.0, rewards=[])

    total = sum(scores.values())
    avg = total / len(scores) if scores else 0.0
    print(f"\nAverage score: {avg:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
