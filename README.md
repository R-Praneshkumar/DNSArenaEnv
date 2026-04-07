---
title: DNS-Env
emoji: "\U0001F310"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - dns
  - infrastructure
  - devops
pinned: false
---

# DNS-Env: DNS Zone File Debugging Environment

## Overview

DNS-Env is a reinforcement learning environment where AI agents learn to debug and configure DNS zone files. Built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology**, it provides a realistic simulation of DNS administration tasks ranging from fixing broken records to configuring complex multi-zone delegations.

## Motivation

- **"It's always DNS"** -- DNS misconfiguration is the #1 cause of internet outages. Yet there is no structured RL environment for training agents to diagnose and repair DNS issues.
- **80% of domains have misconfigured DMARC records** -- Email authentication (SPF, DKIM, DMARC) is critical for deliverability and security, but remains error-prone even for experienced engineers.
- **No existing RL environment or agent benchmark covers DNS debugging** -- DNS-Env fills this gap with grounded, deterministic tasks that mirror real-world operational scenarios.

## Environment Description

The environment simulates DNS zone file editing and validation. An agent is presented with one or more zone files containing deliberate misconfigurations. The agent must use the available commands to inspect, diagnose, and repair the zone files, then submit for grading.

## Action Space

| Command | Arguments | Description |
|---------|-----------|-------------|
| `view_zone` | `zone` (optional) | View current zone file with indexed records |
| `add_record` | `zone`, `name`, `rtype`, `rdata`, `ttl` (optional) | Add a new DNS record |
| `edit_record` | `zone`, `index`, `name`/`rtype`/`rdata`/`ttl` (partial) | Edit record at index |
| `delete_record` | `zone`, `index` | Delete record at index (cannot delete SOA) |
| `check_zone` | `zone` (optional) | Validate zone file for errors |
| `dig` | `qname`, `qtype`, `zone` (optional) | Simulate DNS resolution |
| `submit` | (none) | Submit for grading |

## Observation Space

Each step returns an observation containing:

- **output** -- Command result text (zone file contents, validation errors, dig results, etc.)
- **task_description** -- Current task description and objectives
- **zone_names** -- List of available zone files
- **available_commands** -- Available actions the agent can take
- **done** -- Whether the episode is complete
- **reward** -- Score (0.0--1.0) returned on submission

## Tasks

### Task 1: fix_single_record (Easy)

Fix broken records in an `example.com` zone file. Bugs include missing trailing dots on FQDNs, invalid IP addresses, and malformed MX records.

- **Expected difficulty:** Easy
- **Max steps:** 15

### Task 2: configure_mail (Medium)

Configure complete email delivery for `acme.co`: MX records, SPF, DKIM, and DMARC from a natural language specification.

- **Expected difficulty:** Medium
- **Max steps:** 25

### Task 3: debug_delegation (Hard)

Debug a broken DNS delegation between `parent.org` and `dev.parent.org` across two zone files. Fix NS records, glue records, and SOA serial numbers.

- **Expected difficulty:** Hard
- **Max steps:** 30

## Reward Function

```
score = 0.3 * (structural_validity) + 0.5 * (resolution_correctness) + 0.2 * (no_regressions)
```

| Component | Weight | Description |
|-----------|--------|-------------|
| Structural validity | 0.3 | Zone file has no validation errors |
| Resolution correctness | 0.5 | Required DNS queries resolve correctly |
| No regressions | 0.2 | Pre-existing correct records are preserved |

## Setup & Usage

### Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t dns-env .
docker run -p 7860:7860 dns-env
```

### API Usage

```python
import requests

BASE = "http://localhost:7860"

# Reset with a specific task
obs = requests.post(f"{BASE}/reset", json={"options": {"task_id": "fix_single_record"}}).json()

# View the zone file
obs = requests.post(f"{BASE}/step", json={"action": {"command": "view_zone", "args": {}}}).json()
print(obs["output"])

# Edit a record
obs = requests.post(f"{BASE}/step", json={
    "action": {"command": "edit_record", "args": {"index": 4, "rdata": "example.com."}}
}).json()

# Submit for grading
obs = requests.post(f"{BASE}/step", json={"action": {"command": "submit", "args": {}}}).json()
print(f"Score: {obs['reward']}")
```

## Baseline Scores

| Task | Score | Steps |
|------|-------|-------|
| fix_single_record | ~0.85 | 5--8 |
| configure_mail | ~0.50 | 10--15 |
| debug_delegation | ~0.20 | 15--25 |

## Running the Baseline

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-key"
export ENV_URL="http://localhost:7860"
python inference.py
```

## Technical Details

- Built with **FastAPI** + **Pydantic** for typed request/response models
- Pure Python DNS validation -- no external DNS tools (bind-utils, dig) required
- Deterministic grading via zone file parsing and record matching
- Supports concurrent sessions via `session_id` parameter
- Runs on 2 vCPU / 8 GB RAM within 20 minutes

## OpenEnv Spec Compliance

- step(action) returns observation, reward, done
- reset() returns initial observation
- state() returns episode metadata
- openenv.yaml with spec_version 1
- Typed Pydantic models for all request/response schemas
- Containerized with Docker
- Deployed to HuggingFace Spaces
