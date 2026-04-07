"""FastAPI application for the DNS-Env OpenEnv environment.

Exposes the DNS zone-file debugging environment over HTTP so that
remote agents (or a HuggingFace Spaces front-end) can interact with it
via the standard OpenEnv ``/reset``, ``/step``, ``/state`` endpoints.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# Dual-import pattern -- works as a sub-package *and* when run directly.
# ---------------------------------------------------------------------------

try:
    from .dns_environment import DNSEnvironment
except ImportError:
    from dns_environment import DNSEnvironment  # type: ignore[no-redef]

try:
    from ..models import Action, Observation, State
except ImportError:
    from models import Action, Observation, State  # type: ignore[no-redef]

try:
    from .tasks import TASK_IDS
except ImportError:
    from tasks import TASK_IDS  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DNS-Env",
    description="DNS Zone File Debugging Environment for OpenEnv",
    version="0.1.0",
)

# Allow all origins so HuggingFace Spaces (and other frontends) can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

environments: dict[str, DNSEnvironment] = {}


def get_env(session_id: str = "default") -> DNSEnvironment:
    """Return the environment for *session_id*, creating one if needed."""
    if session_id not in environments:
        environments[session_id] = DNSEnvironment()
    return environments[session_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_LANDING_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DNS-Env | OpenEnv</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0a0e17;color:#c9d1d9;min-height:100vh;overflow-x:hidden}
.glow{position:fixed;width:600px;height:600px;border-radius:50%;filter:blur(150px);opacity:.12;pointer-events:none;z-index:0}
.glow-1{background:#22d3ee;top:-200px;left:-100px}
.glow-2{background:#6366f1;bottom:-200px;right:-100px}
.container{max-width:1100px;margin:0 auto;padding:0 24px;position:relative;z-index:1}
header{padding:40px 0 20px;text-align:center}
.badge{display:inline-block;background:#16a34a;color:#fff;font-size:11px;font-weight:700;padding:4px 14px;border-radius:20px;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:18px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.7}}
h1{font-size:3.2rem;font-weight:800;background:linear-gradient(135deg,#22d3ee,#6366f1,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}
.subtitle{font-size:1.15rem;color:#8b949e;max-width:650px;margin:0 auto 10px}
.tag-row{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin:18px 0 30px}
.tag{background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.3);color:#a5b4fc;font-size:12px;padding:4px 14px;border-radius:20px;font-weight:500}
.section-title{font-size:1.5rem;font-weight:700;color:#e6edf3;margin:40px 0 20px;display:flex;align-items:center;gap:10px}
.section-title span{font-size:1.5rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-bottom:30px}
.card{background:rgba(22,27,40,.8);border:1px solid rgba(99,102,241,.15);border-radius:16px;padding:28px;transition:transform .2s,border-color .2s}
.card:hover{transform:translateY(-4px);border-color:rgba(99,102,241,.4)}
.card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.card h3{font-size:1.1rem;color:#e6edf3}
.diff{font-size:11px;font-weight:700;padding:3px 12px;border-radius:12px;text-transform:uppercase;letter-spacing:.5px}
.diff-easy{background:rgba(34,197,94,.15);color:#4ade80;border:1px solid rgba(34,197,94,.3)}
.diff-medium{background:rgba(234,179,8,.15);color:#facc15;border:1px solid rgba(234,179,8,.3)}
.diff-hard{background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.3)}
.card p{color:#8b949e;font-size:.9rem;line-height:1.6}
.card code{background:rgba(99,102,241,.1);color:#a5b4fc;padding:2px 6px;border-radius:4px;font-size:.82rem}
.reward-box{background:rgba(22,27,40,.8);border:1px solid rgba(99,102,241,.15);border-radius:16px;padding:28px;margin-bottom:30px}
.reward-bar{display:flex;align-items:center;gap:12px;margin:12px 0}
.reward-label{width:140px;font-size:.85rem;color:#8b949e}
.bar-track{flex:1;height:24px;background:rgba(255,255,255,.05);border-radius:12px;overflow:hidden;position:relative}
.bar-fill{height:100%;border-radius:12px;display:flex;align-items:center;justify-content:flex-end;padding-right:10px;font-size:11px;font-weight:700;color:#fff;transition:width .6s ease}
.bar-30{width:30%;background:linear-gradient(90deg,#6366f1,#818cf8)}
.bar-50{width:50%;background:linear-gradient(90deg,#22d3ee,#06b6d4)}
.bar-20{width:20%;background:linear-gradient(90deg,#ec4899,#f472b6)}
.api-table{width:100%;border-collapse:collapse;margin-bottom:30px}
.api-table th{text-align:left;padding:12px 16px;background:rgba(99,102,241,.08);color:#a5b4fc;font-size:.8rem;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid rgba(99,102,241,.15)}
.api-table td{padding:12px 16px;border-bottom:1px solid rgba(255,255,255,.04);font-size:.9rem}
.api-table tr:hover td{background:rgba(99,102,241,.04)}
.method{font-weight:700;font-size:.8rem;padding:3px 10px;border-radius:6px;font-family:monospace}
.method-get{background:rgba(34,197,94,.15);color:#4ade80}
.method-post{background:rgba(234,179,8,.15);color:#facc15}
.endpoint{font-family:'JetBrains Mono',monospace;color:#22d3ee}
.demo-box{background:rgba(22,27,40,.8);border:1px solid rgba(99,102,241,.15);border-radius:16px;padding:28px;margin-bottom:30px}
.demo-btn{background:linear-gradient(135deg,#6366f1,#4f46e5);color:#fff;border:none;padding:10px 28px;border-radius:10px;font-size:.95rem;font-weight:600;cursor:pointer;transition:transform .15s,box-shadow .15s}
.demo-btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(99,102,241,.3)}
.demo-btn:active{transform:translateY(0)}
#demo-output{margin-top:16px;background:#0d1117;border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:16px;font-family:'JetBrains Mono',monospace;font-size:.82rem;color:#7ee787;max-height:400px;overflow-y:auto;white-space:pre-wrap;display:none}
footer{text-align:center;padding:40px 0;color:#484f58;font-size:.82rem;border-top:1px solid rgba(255,255,255,.04);margin-top:20px}
footer a{color:#6366f1;text-decoration:none}
.spec-badges{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin:16px 0}
.spec-badge{display:flex;align-items:center;gap:5px;background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.2);color:#4ade80;font-size:12px;padding:4px 12px;border-radius:8px}
</style>
</head>
<body>
<div class="glow glow-1"></div>
<div class="glow glow-2"></div>
<div class="container">
<header>
<div class="badge">Running</div>
<h1>DNS-Env</h1>
<p class="subtitle">DNS Zone File Debugging Environment for OpenEnv &mdash; Train AI agents to diagnose and fix real-world DNS misconfigurations.</p>
<div class="tag-row">
<span class="tag">openenv</span>
<span class="tag">reinforcement-learning</span>
<span class="tag">dns</span>
<span class="tag">infrastructure</span>
<span class="tag">devops</span>
</div>
</header>

<div class="section-title"><span>&#127919;</span> Tasks</div>
<div class="cards">
<div class="card">
<div class="card-header"><h3>fix_single_record</h3><span class="diff diff-easy">Easy</span></div>
<p>Fix broken records in <code>example.com</code> &mdash; missing trailing dots, invalid IPs, malformed MX. The classic DNS gotchas that cause real outages.</p>
</div>
<div class="card">
<div class="card-header"><h3>configure_mail</h3><span class="diff diff-medium">Medium</span></div>
<p>Set up complete email delivery for <code>acme.co</code> &mdash; MX records, SPF authorization, DMARC policy. 80% of domains get this wrong.</p>
</div>
<div class="card">
<div class="card-header"><h3>debug_delegation</h3><span class="diff diff-hard">Hard</span></div>
<p>Repair broken NS delegation across <code>parent.org</code> and <code>dev.parent.org</code> &mdash; fix glue records, NS consistency, SOA serials.</p>
</div>
</div>

<div class="section-title"><span>&#9878;</span> Reward Function</div>
<div class="reward-box">
<div class="reward-bar"><span class="reward-label">Structural validity</span><div class="bar-track"><div class="bar-fill bar-30">30%</div></div></div>
<div class="reward-bar"><span class="reward-label">Resolution checks</span><div class="bar-track"><div class="bar-fill bar-50">50%</div></div></div>
<div class="reward-bar"><span class="reward-label">No regressions</span><div class="bar-track"><div class="bar-fill bar-20">20%</div></div></div>
<p style="color:#8b949e;font-size:.85rem;margin-top:14px">Deterministic grading via zone-file parsing and record matching. Partial credit for incremental progress.</p>
</div>

<div class="section-title"><span>&#128268;</span> API Endpoints</div>
<table class="api-table">
<thead><tr><th>Method</th><th>Endpoint</th><th>Description</th></tr></thead>
<tbody>
<tr><td><span class="method method-get">GET</span></td><td class="endpoint">/health</td><td>Liveness probe</td></tr>
<tr><td><span class="method method-post">POST</span></td><td class="endpoint">/reset</td><td>Start a new episode with a task</td></tr>
<tr><td><span class="method method-post">POST</span></td><td class="endpoint">/step</td><td>Execute an action (view, edit, dig, submit)</td></tr>
<tr><td><span class="method method-get">GET</span></td><td class="endpoint">/state</td><td>Current episode state</td></tr>
<tr><td><span class="method method-get">GET</span></td><td class="endpoint">/tasks</td><td>List available tasks</td></tr>
<tr><td><span class="method method-get">GET</span></td><td class="endpoint"><a href="/docs" style="color:#22d3ee">/docs</a></td><td>Interactive Swagger UI</td></tr>
</tbody>
</table>

<div class="section-title"><span>&#9889;</span> Live Demo</div>
<div class="demo-box">
<p style="color:#8b949e;margin-bottom:14px">Click to run a live interaction against this environment:</p>
<button class="demo-btn" onclick="runDemo()">Run Demo &rarr; Reset + View + Check</button>
<div id="demo-output"></div>
</div>

<div class="section-title"><span>&#9989;</span> OpenEnv Spec Compliance</div>
<div class="spec-badges">
<span class="spec-badge">&#10003; step(action)</span>
<span class="spec-badge">&#10003; reset()</span>
<span class="spec-badge">&#10003; state()</span>
<span class="spec-badge">&#10003; openenv.yaml</span>
<span class="spec-badge">&#10003; Typed Pydantic Models</span>
<span class="spec-badge">&#10003; Docker Container</span>
<span class="spec-badge">&#10003; 3 Graded Tasks</span>
<span class="spec-badge">&#10003; Baseline Inference</span>
</div>

<footer>
<p>DNS-Env &mdash; Meta PyTorch OpenEnv Hackathon x Scaler School of Technology</p>
<p style="margin-top:6px"><a href="/docs">API Docs</a> &nbsp;&bull;&nbsp; <a href="https://github.com/meta-pytorch/OpenEnv">OpenEnv Framework</a></p>
</footer>
</div>
<script>
async function runDemo(){
const o=document.getElementById('demo-output');o.style.display='block';o.textContent='Resetting environment (fix_single_record)...\\n';
try{
let r=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({options:{task_id:'fix_single_record'}})});
let d=await r.json();o.textContent+='\\n--- RESET OK ---\\n';o.textContent+='Task: '+d.task_description.substring(0,150)+'...\\nZones: '+d.zone_names.join(', ')+'\\n';
o.textContent+='\\nViewing zone file...\\n';
r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:{command:'view_zone',args:{}}})});
d=await r.json();o.textContent+='\\n'+d.output+'\\n';
o.textContent+='\\nRunning zone validation...\\n';
r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:{command:'check_zone',args:{}}})});
d=await r.json();o.textContent+='\\n'+d.output+'\\n';
o.textContent+='\\n--- DEMO COMPLETE ---\\nThe agent would now fix these errors and submit for grading.';
}catch(e){o.textContent+='\\nError: '+e.message;}
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with environment info and live demo."""
    return _LANDING_HTML


@app.get("/health")
async def health() -> dict:
    """Liveness / readiness probe."""
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: dict = {}) -> dict:  # noqa: B006 -- mutable default is fine for FastAPI body
    """Reset the environment and start a new episode.

    Optional body fields
    --------------------
    session_id : str
        Identifies the caller's session (default ``"default"``).
    seed : int | None
        RNG seed for reproducibility.
    episode_id : str | None
        Caller-supplied episode identifier.
    options : dict
        May contain ``task_id`` (e.g. ``"fix_single_record"``).
    """
    session_id: str = request.get("session_id", "default")
    env = get_env(session_id)
    obs: Observation = env.reset(
        seed=request.get("seed"),
        episode_id=request.get("episode_id"),
        options=request.get("options", {}),
    )
    return obs.model_dump()


@app.post("/step")
async def step(request: dict) -> dict:
    """Execute one agent action and return the observation.

    Body fields
    -----------
    session_id : str
        Session identifier (default ``"default"``).
    action : dict
        Must contain ``command`` (str) and optionally ``args`` (dict)
        and ``metadata`` (dict).  If the top-level dict already has a
        ``command`` key and no ``action`` wrapper, it is treated as the
        action directly for convenience.
    """
    session_id: str = request.get("session_id", "default")

    # Accept either {"action": {…}} or a flat {command, args, …} body.
    action_data = request.get("action", request)
    action = Action(**action_data)

    env = get_env(session_id)
    obs: Observation = env.step(action)
    return obs.model_dump()


@app.get("/state")
async def get_state(session_id: str = "default") -> dict:
    """Return the current episode state (step count, task id, etc.)."""
    env = get_env(session_id)
    return env.state.model_dump()


@app.get("/tasks")
async def list_tasks() -> dict:
    """List the available task identifiers."""
    return {"tasks": list(TASK_IDS)}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the DNS-Env server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
