#!/usr/bin/env python3
"""Baseline inference script for the DNS-Env OpenEnv hackathon.

Runs an LLM-driven agent through all three DNS debugging tasks,
communicating with the environment server over HTTP.

Requirements
------------
- Environment server running at ENV_URL (default http://localhost:7860).
- OpenAI-compatible LLM endpoint at API_BASE_URL.
- HF_TOKEN or OPENAI_API_KEY set for authentication.

Stdout logging format
---------------------
[START] task_id=<id>
[STEP]  task_id=<id> step=<n> command=<cmd> reward=<r> done=<bool>
[END]   task_id=<id> score=<float> steps=<n>
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from typing import Any

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

# Safety limits
MAX_STEPS_PER_TASK = 25
MAX_RETRIES_HTTP = 3
HTTP_TIMEOUT = 60  # seconds

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", ""),
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
# HTTP helpers (self-contained, no dependency on client.py)
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
                print(
                    f"[ERROR] HTTP POST {endpoint} failed after "
                    f"{MAX_RETRIES_HTTP} attempts: {exc}",
                    file=sys.stderr,
                )
                raise
            time.sleep(1.0 * attempt)
    return {}  # unreachable, satisfies type checker


def reset_env(task_id: str) -> dict[str, Any]:
    """Reset the environment for a specific task."""
    body: dict[str, Any] = {
        "session_id": "default",
        "options": {"task_id": task_id},
    }
    return _post("/reset", body)


def step_env(action: dict[str, Any]) -> dict[str, Any]:
    """Execute one action in the environment."""
    body = {
        "session_id": "default",
        "action": action,
    }
    return _post("/step", body)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    obs: dict[str, Any],
    task_id: str,
    step_num: int,
    max_steps: int,
    history: list[dict[str, str]],
) -> str:
    """Build the user-turn prompt from the current observation.

    Includes the task description, current observation output, zone names,
    step budget, and recent action history for context.
    """
    parts: list[str] = []

    # Task description
    task_desc = obs.get("task_description", "")
    if task_desc:
        parts.append(f"## Task: {task_id}\n{task_desc}")

    # Available zones
    zone_names = obs.get("zone_names", [])
    if zone_names:
        parts.append(f"Available zones: {', '.join(zone_names)}")

    # Step budget
    remaining = max_steps - step_num
    parts.append(f"Step {step_num}/{max_steps} (remaining: {remaining})")

    # Recent action history (last 3 actions for context)
    if history:
        recent = history[-3:]
        history_lines = []
        for h in recent:
            history_lines.append(f"  Action: {h['action']}")
            # Truncate long outputs
            result_preview = h["result"][:300]
            if len(h["result"]) > 300:
                result_preview += "..."
            history_lines.append(f"  Result: {result_preview}")
        parts.append("## Recent History\n" + "\n".join(history_lines))

    # Current observation output
    output = obs.get("output", "")
    if output:
        parts.append(f"## Current Output\n{output}")

    # Instruction
    if remaining <= 3:
        parts.append(
            "WARNING: You are running low on steps. If your fixes are "
            "done, submit now with: {\"command\": \"submit\", \"args\": {}}"
        )

    parts.append(
        "Respond with a single JSON object: "
        "{\"command\": \"...\", \"args\": {...}}"
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

# Patterns for extracting JSON from LLM responses
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def parse_llm_response(text: str | None) -> dict[str, Any]:
    """Extract a JSON action from the LLM's response text.

    Handles several formats:
    1. Pure JSON response.
    2. JSON inside markdown code blocks (```json ... ```).
    3. JSON embedded in surrounding text.
    4. Fallback: returns a safe default action (view_zone).

    Returns
    -------
    dict
        Action dict with "command" (str) and "args" (dict).
    """
    default_action: dict[str, Any] = {"command": "view_zone", "args": {}}

    if not text:
        return default_action

    text = text.strip()

    # Strategy 1: try parsing the entire response as JSON
    action = _try_parse_json(text)
    if action is not None:
        return action

    # Strategy 2: look for JSON inside markdown code blocks
    match = _JSON_BLOCK_RE.search(text)
    if match:
        action = _try_parse_json(match.group(1).strip())
        if action is not None:
            return action

    # Strategy 3: find the first JSON object in the text
    match = _JSON_OBJECT_RE.search(text)
    if match:
        action = _try_parse_json(match.group(0))
        if action is not None:
            return action

    # Strategy 4: fallback
    print(
        f"[WARN] Could not parse LLM response as JSON, using fallback. "
        f"Response was: {text[:200]}",
        file=sys.stderr,
    )
    return default_action


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to parse *text* as a JSON action. Returns None on failure."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "command" in data:
            # Ensure args is always a dict
            if "args" not in data or not isinstance(data.get("args"), dict):
                data["args"] = data.get("args", {}) or {}
            return {"command": str(data["command"]), "args": data["args"]}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def call_llm(
    messages: list[dict[str, str]],
    temperature: float = 0.0,
) -> str:
    """Call the LLM and return the assistant's response text.

    Retries once on transient errors.
    """
    for attempt in range(1, 3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as exc:
            if attempt == 2:
                print(
                    f"[ERROR] LLM call failed: {exc}",
                    file=sys.stderr,
                )
                raise
            time.sleep(2.0)
    return ""  # unreachable


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


def run_task(task_id: str) -> float:
    """Run a single task and return the score.

    Returns
    -------
    float
        The final score for the task (0.0-1.0).
    """
    print(f"[START] task_id={task_id}")

    # Reset environment
    try:
        obs = reset_env(task_id)
    except Exception as exc:
        print(f"[ERROR] Failed to reset environment for {task_id}: {exc}", file=sys.stderr)
        print(f"[END] task_id={task_id} score=0.0 steps=0")
        return 0.0

    step_num = 0
    max_steps = MAX_STEPS_PER_TASK
    history: list[dict[str, str]] = []

    # Build conversation with system prompt
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while not obs.get("done", False):
        step_num += 1

        # Build the user prompt for this step
        prompt = build_prompt(obs, task_id, step_num, max_steps, history)
        messages.append({"role": "user", "content": prompt})

        # Call the LLM
        try:
            llm_text = call_llm(messages)
        except Exception as exc:
            print(
                f"[ERROR] LLM call failed at step {step_num}: {exc}",
                file=sys.stderr,
            )
            # Try to submit gracefully
            action = {"command": "submit", "args": {}}
            llm_text = json.dumps(action)

        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": llm_text})

        # Parse the action
        action = parse_llm_response(llm_text)

        # Record in history
        history.append({"action": json.dumps(action), "result": ""})

        # Execute the action
        try:
            obs = step_env(action)
        except Exception as exc:
            print(
                f"[ERROR] step_env failed at step {step_num}: {exc}",
                file=sys.stderr,
            )
            # Force submit on error
            try:
                obs = step_env({"command": "submit", "args": {}})
            except Exception:
                obs = {"done": True, "reward": 0.0}
            break

        # Update history with result
        if history:
            output_preview = obs.get("output", "")[:500]
            history[-1]["result"] = output_preview

        # Log the step
        reward = obs.get("reward")
        reward_str = f"{reward}" if reward is not None else "null"
        done = obs.get("done", False)
        print(
            f"[STEP] task_id={task_id} step={step_num} "
            f"command={action['command']} reward={reward_str} done={done}"
        )

        # Safety limit: force submit
        if step_num >= max_steps and not obs.get("done", False):
            print(
                f"[WARN] Reached step limit ({max_steps}) for {task_id}, "
                f"forcing submit.",
                file=sys.stderr,
            )
            try:
                obs = step_env({"command": "submit", "args": {}})
                reward = obs.get("reward")
                reward_str = f"{reward}" if reward is not None else "null"
                print(
                    f"[STEP] task_id={task_id} step={step_num + 1} "
                    f"command=submit reward={reward_str} done={obs.get('done', False)}"
                )
            except Exception:
                obs = {"done": True, "reward": 0.0}
            break

        # Keep conversation history manageable: trim old messages if too long
        # Keep system prompt + last 20 exchanges (40 messages)
        if len(messages) > 41:
            messages = [messages[0]] + messages[-40:]

    # Final score
    score = obs.get("reward", 0.0)
    if score is None:
        score = 0.0
    print(f"[END] task_id={task_id} score={score} steps={step_num}")
    return float(score)


def main() -> None:
    """Run inference on all tasks and report aggregate results."""
    print("=" * 60)
    print("DNS-Env Baseline Inference")
    print(f"  API_BASE_URL: {API_BASE_URL}")
    print(f"  MODEL_NAME:   {MODEL_NAME}")
    print(f"  ENV_URL:      {ENV_URL}")
    print("=" * 60)

    # Verify the environment server is reachable
    try:
        resp = requests.get(f"{ENV_URL}/health", timeout=10)
        resp.raise_for_status()
        print(f"Environment server health: {resp.json()}")
    except Exception as exc:
        print(
            f"[FATAL] Cannot reach environment server at {ENV_URL}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Run all tasks
    scores: dict[str, float] = {}
    for task_id in TASKS:
        try:
            score = run_task(task_id)
            scores[task_id] = score
        except Exception as exc:
            print(
                f"[ERROR] Task {task_id} failed with exception: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            scores[task_id] = 0.0
            print(f"[END] task_id={task_id} score=0.0 steps=0")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    total = 0.0
    for task_id, score in scores.items():
        print(f"  {task_id:25s}  {score:.4f}")
        total += score
    avg = total / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':25s}  {avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
