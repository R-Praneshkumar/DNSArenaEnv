"""HTTP client for the DNS-Env OpenEnv environment."""
from __future__ import annotations

import requests
from typing import Any


class DNSEnvClient:
    """Simple HTTP client for the DNS environment server.

    Wraps the REST endpoints exposed by the FastAPI server so callers
    can interact with the environment from plain Python without manually
    assembling HTTP requests.

    Parameters
    ----------
    base_url:
        Root URL of the environment server (default ``http://localhost:7860``).
    session_id:
        Identifies the caller's session. The server maintains independent
        environment state per session.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        session_id: str = "default",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Liveness check. Returns ``{"status": "ok"}`` when the server is up."""
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        task_id: str | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> dict:
        """Reset the environment and start a new episode.

        Parameters
        ----------
        task_id:
            One of ``fix_single_record``, ``configure_mail``,
            ``debug_delegation``.  When *None* the server cycles tasks.
        seed:
            Optional RNG seed for reproducibility.
        episode_id:
            Optional caller-supplied episode identifier.

        Returns
        -------
        dict
            Observation JSON with keys: ``output``, ``task_description``,
            ``zone_names``, ``available_commands``, ``done``, ``reward``,
            ``metadata``.
        """
        body: dict[str, Any] = {"session_id": self.session_id}
        options: dict[str, Any] = {}
        if task_id:
            options["task_id"] = task_id
        if options:
            body["options"] = options
        if seed is not None:
            body["seed"] = seed
        if episode_id is not None:
            body["episode_id"] = episode_id
        resp = requests.post(
            f"{self.base_url}/reset", json=body, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, command: str, **args: Any) -> dict:
        """Execute one action in the environment.

        Parameters
        ----------
        command:
            One of the available commands (``view_zone``, ``add_record``,
            ``edit_record``, ``delete_record``, ``check_zone``, ``dig``,
            ``submit``).
        **args:
            Keyword arguments forwarded as the action's ``args`` dict.

        Returns
        -------
        dict
            Observation JSON.
        """
        body = {
            "session_id": self.session_id,
            "action": {"command": command, "args": args},
        }
        resp = requests.post(
            f"{self.base_url}/step", json=body, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        """Return the current episode state (step count, task id, etc.)."""
        resp = requests.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> list[str]:
        """List available task identifiers."""
        resp = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["tasks"]
