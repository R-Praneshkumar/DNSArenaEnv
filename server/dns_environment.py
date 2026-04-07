"""DNS zone-file debugging environment for the OpenEnv hackathon.

Provides ``DNSEnvironment`` -- a stateful, step/reset/state RL environment
where an agent diagnoses and fixes broken DNS zone files.
"""

from __future__ import annotations

import copy
import random
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Dual-import pattern so the module works both as part of the package
# (``from dns_env.server.dns_environment import …``) and when executed
# directly (``python dns_environment.py``).
# ---------------------------------------------------------------------------

try:
    from .dns_utils import (
        DNSRecord,
        render_zone_file,
        validate_zone,
        simulate_dig,
        grade_zone,
    )
    from .tasks import get_task, TASK_IDS
except ImportError:
    from dns_utils import (  # type: ignore[no-redef]
        DNSRecord,
        render_zone_file,
        validate_zone,
        simulate_dig,
        grade_zone,
    )
    from tasks import get_task, TASK_IDS  # type: ignore[no-redef]

try:
    from ..models import Action, Observation, State
except ImportError:
    from models import Action, Observation, State  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_zone_indexed(records: list[DNSRecord], origin: str) -> str:
    """Render a zone file with ``[N]`` index comments for each record.

    The indices let the agent reference records by number when editing or
    deleting.
    """
    lines: list[str] = []
    lines.append(f"$ORIGIN {origin}.")
    # Determine default TTL from SOA if available
    default_ttl = 86400
    if records:
        first = records[0]
        if first.ttl is not None:
            default_ttl = first.ttl
    lines.append(f"$TTL {default_ttl}")
    lines.append("")
    for idx, rec in enumerate(records):
        ttl_str = f"{rec.ttl}" if rec.ttl is not None else ""
        parts = [rec.name, ttl_str, rec.rclass, rec.rtype, rec.rdata]
        line = "\t".join(p for p in parts if p)
        lines.append(f"{line}  ; [{idx}]")
    return "\n".join(lines)


def _resolve_zone(zones: dict[str, list[DNSRecord]], requested: str | None) -> tuple[str, list[DNSRecord] | None]:
    """Return ``(zone_name, records)`` for the requested zone.

    If *requested* is ``None``, the first zone is used.  Returns
    ``(zone_name, None)`` when the zone cannot be found.
    """
    if not zones:
        return ("", None)
    if requested is None:
        zone_name = next(iter(zones))
    else:
        # Normalise: strip trailing dot if present
        zone_name = requested.rstrip(".")
    records = zones.get(zone_name)
    return (zone_name, records)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DNSEnvironment:
    """Stateful DNS zone-file debugging environment.

    Lifecycle
    ---------
    1. ``reset()`` -- load a task, populate ``self.zones``.
    2. ``step(action)`` -- process agent commands, return observations.
    3. ``state`` -- read-only snapshot of episode metadata.
    """

    VALID_COMMANDS: list[str] = [
        "view_zone",
        "add_record",
        "edit_record",
        "delete_record",
        "check_zone",
        "dig",
        "submit",
    ]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self.zones: dict[str, list[DNSRecord]] = {}
        self.task_config: dict[str, Any] = {}
        self.episode_id: str | None = None
        self.step_count: int = 0
        self._done: bool = False
        self._last_reward: float | None = None
        self._task_cycle_index: int = 0

    # ------------------------------------------------------------------
    # reset / step / state
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> Observation:
        """Reset the environment and load a new task.

        Parameters
        ----------
        seed:
            Optional RNG seed for reproducibility.
        episode_id:
            Optional caller-supplied episode identifier.  A UUID is
            generated when not provided.
        options:
            Dict that may contain ``task_id`` (one of
            :pydata:`TASK_IDS`).  When omitted the environment cycles
            through the available tasks round-robin.
        """
        if seed is not None:
            random.seed(seed)

        options = options or {}
        task_id: str | None = options.get("task_id")

        # Pick a task -------------------------------------------------
        if task_id is None:
            task_id = TASK_IDS[self._task_cycle_index % len(TASK_IDS)]
            self._task_cycle_index += 1

        self.task_config = get_task(task_id)

        # Deep-copy zones so original task data stays pristine ---------
        self.zones = {
            name: copy.deepcopy(records)
            for name, records in self.task_config["zones"].items()
        }

        # Episode bookkeeping -----------------------------------------
        self.episode_id = episode_id or uuid.uuid4().hex
        self.step_count = 0
        self._done = False
        self._last_reward = None

        return Observation(
            output=(
                f"Episode {self.episode_id} started.\n\n"
                f"Task: {self.task_config['description']}\n\n"
                f"Available zones: {', '.join(self.zones.keys())}\n"
                "Use 'view_zone' to inspect a zone, then fix any issues and 'submit' when ready."
            ),
            task_description=self.task_config["description"],
            zone_names=list(self.zones.keys()),
            done=False,
        )

    def step(self, action: Action) -> Observation:
        """Process *action* and return the resulting observation.

        If the episode is already done, further steps return a terminal
        observation.  When ``step_count`` reaches ``max_steps`` the
        environment auto-submits.
        """
        if not self.task_config:
            return Observation(
                output="Error: environment has not been reset. Call /reset first.",
                done=False,
            )

        if self._done:
            return Observation(
                output="Episode is already done. Call /reset to start a new episode.",
                done=True,
                reward=self._last_reward,
                zone_names=list(self.zones.keys()),
                task_description=self.task_config.get("description", ""),
            )

        self.step_count += 1
        max_steps: int = self.task_config.get("max_steps", 30)

        # Auto-submit when budget exhausted ----------------------------
        if self.step_count >= max_steps and action.command != "submit":
            auto_obs = self._handle_submit({})
            auto_obs.output = (
                f"Step limit ({max_steps}) reached -- auto-submitting.\n\n"
                + auto_obs.output
            )
            return auto_obs

        # Dispatch to handler ------------------------------------------
        command = action.command.strip().lower()
        handler = {
            "view_zone": self._handle_view_zone,
            "add_record": self._handle_add_record,
            "edit_record": self._handle_edit_record,
            "delete_record": self._handle_delete_record,
            "check_zone": self._handle_check_zone,
            "dig": self._handle_dig,
            "submit": self._handle_submit,
        }.get(command)

        if handler is None:
            return self._obs(
                f"Error: unknown command '{action.command}'. "
                f"Available commands: {', '.join(self.VALID_COMMANDS)}"
            )

        try:
            return handler(action.args)
        except Exception as exc:  # pragma: no cover -- safety net
            return self._obs(f"Error processing '{command}': {exc}")

    @property
    def state(self) -> State:
        """Read-only snapshot of the current episode state."""
        return State(
            episode_id=self.episode_id,
            step_count=self.step_count,
            task_id=self.task_config.get("task_id", ""),
            max_steps=self.task_config.get("max_steps", 30),
        )

    # ------------------------------------------------------------------
    # Command handlers (private)
    # ------------------------------------------------------------------

    def _handle_view_zone(self, args: dict[str, Any]) -> Observation:
        zone_name, records = _resolve_zone(self.zones, args.get("zone"))
        if records is None:
            return self._obs(self._zone_not_found_msg(args.get("zone")))
        rendered = _render_zone_indexed(records, zone_name)
        return self._obs(f"Zone: {zone_name}\n\n{rendered}")

    # -- add_record ----------------------------------------------------

    def _handle_add_record(self, args: dict[str, Any]) -> Observation:
        zone_name, records = _resolve_zone(self.zones, args.get("zone"))
        if records is None:
            return self._obs(self._zone_not_found_msg(args.get("zone")))

        # Validate required fields
        missing = [f for f in ("name", "rtype", "rdata") if f not in args]
        if missing:
            return self._obs(
                f"Error: add_record requires args: name, rtype, rdata. "
                f"Missing: {', '.join(missing)}"
            )

        new_record = DNSRecord(
            name=str(args["name"]),
            rtype=str(args["rtype"]).upper(),
            rdata=str(args["rdata"]),
            ttl=int(args["ttl"]) if "ttl" in args else None,
            rclass=str(args.get("rclass", "IN")),
        )
        records.append(new_record)

        rendered = _render_zone_indexed(records, zone_name)
        return self._obs(
            f"Record added to {zone_name} at index [{len(records) - 1}].\n\n{rendered}"
        )

    # -- edit_record ---------------------------------------------------

    def _handle_edit_record(self, args: dict[str, Any]) -> Observation:
        zone_name, records = _resolve_zone(self.zones, args.get("zone"))
        if records is None:
            return self._obs(self._zone_not_found_msg(args.get("zone")))

        if "index" not in args:
            return self._obs("Error: edit_record requires 'index' in args.")

        try:
            index = int(args["index"])
        except (ValueError, TypeError):
            return self._obs(f"Error: 'index' must be an integer, got '{args['index']}'.")

        if index < 0 or index >= len(records):
            return self._obs(
                f"Error: index {index} out of range. "
                f"Valid range: 0..{len(records) - 1}."
            )

        rec = records[index]

        # Partial update -- only touch fields the agent supplies
        if "name" in args:
            rec.name = str(args["name"])
        if "rtype" in args:
            rec.rtype = str(args["rtype"]).upper()
        if "rdata" in args:
            rec.rdata = str(args["rdata"])
        if "ttl" in args:
            rec.ttl = int(args["ttl"]) if args["ttl"] is not None else None
        if "rclass" in args:
            rec.rclass = str(args["rclass"])

        rendered = _render_zone_indexed(records, zone_name)
        return self._obs(
            f"Record [{index}] in {zone_name} updated.\n\n{rendered}"
        )

    # -- delete_record -------------------------------------------------

    def _handle_delete_record(self, args: dict[str, Any]) -> Observation:
        zone_name, records = _resolve_zone(self.zones, args.get("zone"))
        if records is None:
            return self._obs(self._zone_not_found_msg(args.get("zone")))

        if "index" not in args:
            return self._obs("Error: delete_record requires 'index' in args.")

        try:
            index = int(args["index"])
        except (ValueError, TypeError):
            return self._obs(f"Error: 'index' must be an integer, got '{args['index']}'.")

        if index < 0 or index >= len(records):
            return self._obs(
                f"Error: index {index} out of range. "
                f"Valid range: 0..{len(records) - 1}."
            )

        # Protect the SOA record
        if index == 0 and records[0].rtype.upper() == "SOA":
            return self._obs(
                "Error: cannot delete the SOA record (index 0). "
                "Edit it instead if it needs changes."
            )

        deleted = records.pop(index)
        rendered = _render_zone_indexed(records, zone_name)
        return self._obs(
            f"Record [{index}] ({deleted.name} {deleted.rtype} {deleted.rdata}) "
            f"deleted from {zone_name}.\n\n{rendered}"
        )

    # -- check_zone ----------------------------------------------------

    def _handle_check_zone(self, args: dict[str, Any]) -> Observation:
        zone_name, records = _resolve_zone(self.zones, args.get("zone"))
        if records is None:
            return self._obs(self._zone_not_found_msg(args.get("zone")))

        errors = validate_zone(records, zone_name)

        if not errors:
            return self._obs(
                f"Zone validation passed for {zone_name}. No errors found."
            )

        error_lines = "\n".join(
            f"  {i + 1}. {err}" for i, err in enumerate(errors)
        )
        return self._obs(
            f"Zone {zone_name} has {len(errors)} error(s):\n{error_lines}"
        )

    # -- dig -----------------------------------------------------------

    def _handle_dig(self, args: dict[str, Any]) -> Observation:
        qname: str | None = args.get("qname")
        qtype: str | None = args.get("qtype")

        if not qname or not qtype:
            return self._obs(
                "Error: dig requires 'qname' and 'qtype' in args. "
                "Example: {\"qname\": \"www.example.com\", \"qtype\": \"A\"}"
            )

        # Determine which zone to query --------------------------------
        zone_name_arg = args.get("zone")
        if zone_name_arg is not None:
            zone_name, records = _resolve_zone(self.zones, zone_name_arg)
            if records is None:
                return self._obs(self._zone_not_found_msg(zone_name_arg))
        else:
            # Auto-detect zone: pick the zone whose origin is a suffix
            # of qname, preferring the longest match.
            zone_name, records = self._match_zone_for_qname(qname)
            if records is None:
                # Fall back to first zone
                zone_name, records = _resolve_zone(self.zones, None)
                if records is None:
                    return self._obs("Error: no zones available.")

        result = simulate_dig(records, zone_name, str(qname), str(qtype).upper())
        return self._obs(f";; Querying {zone_name} for {qname} {qtype}\n\n{result}")

    # -- submit --------------------------------------------------------

    def _handle_submit(self, args: dict[str, Any]) -> Observation:
        required_checks = self.task_config.get("required_checks", [])
        original_correct_raw = self.task_config.get("original_correct", {})

        # original_correct is a dict {zone_name: [(name, type, rdata), ...]}
        # grade_zone expects a flat list of tuples for a single zone
        if not isinstance(original_correct_raw, dict):
            original_correct_raw = {}

        if len(self.zones) == 1:
            # Single-zone task
            zone_name = next(iter(self.zones))
            records = self.zones[zone_name]
            oc = original_correct_raw.get(zone_name, [])
            result = grade_zone(
                records, zone_name, required_checks, oc or None
            )
            score = result.get("score", 0.0)
            breakdown = self._format_grading(zone_name, result)
        else:
            # Multi-zone task: grade each zone independently and average
            zone_results: dict[str, dict[str, Any]] = {}
            total_score = 0.0
            for zn, recs in self.zones.items():
                # Filter checks relevant to this zone if checks carry a
                # ``zone`` key; otherwise pass all checks to every zone.
                zone_checks = [
                    c for c in required_checks
                    if c.get("zone", zn) == zn
                ]
                # Skip checks that don't have a zone key or qname (like delegation_consistency)
                if not zone_checks:
                    zone_checks = [
                        c for c in required_checks
                        if "zone" not in c and "qname" in c
                    ]
                oc = original_correct_raw.get(zn, [])
                res = grade_zone(recs, zn, zone_checks, oc or None)
                zone_results[zn] = res
                total_score += res.get("score", 0.0)
            score = total_score / max(len(zone_results), 1)
            parts = [
                self._format_grading(zn, res)
                for zn, res in zone_results.items()
            ]
            breakdown = "\n---\n".join(parts)

        self._done = True
        self._last_reward = score

        return Observation(
            output=(
                f"=== Submission Graded ===\n\n"
                f"{breakdown}\n\n"
                f"Final score: {score:.2f}"
            ),
            task_description=self.task_config.get("description", ""),
            zone_names=list(self.zones.keys()),
            done=True,
            reward=score,
        )

    # ------------------------------------------------------------------
    # Utilities (private)
    # ------------------------------------------------------------------

    def _obs(self, output: str) -> Observation:
        """Convenience builder for a non-terminal observation."""
        return Observation(
            output=output,
            task_description=self.task_config.get("description", ""),
            zone_names=list(self.zones.keys()),
            done=False,
        )

    @staticmethod
    def _zone_not_found_msg(requested: str | None) -> str:
        if requested:
            return f"Error: zone '{requested}' not found. Use 'view_zone' to list available zones."
        return "Error: no zones available. Has the environment been reset?"

    def _match_zone_for_qname(self, qname: str) -> tuple[str, list[DNSRecord] | None]:
        """Find the zone whose origin is the longest suffix of *qname*."""
        qname_lower = qname.rstrip(".").lower()
        best_name: str | None = None
        best_len = -1
        for zn in self.zones:
            zn_lower = zn.lower()
            if qname_lower == zn_lower or qname_lower.endswith("." + zn_lower):
                if len(zn_lower) > best_len:
                    best_name = zn
                    best_len = len(zn_lower)
        if best_name is not None:
            return (best_name, self.zones[best_name])
        return ("", None)

    @staticmethod
    def _format_grading(zone_name: str, result: dict[str, Any]) -> str:
        """Format a single zone's grading result into readable text."""
        lines: list[str] = [f"Zone: {zone_name}"]
        lines.append(f"  Score: {result.get('score', 0.0):.2f}")
        lines.append(f"  Passed: {result.get('passed', 0)}, Failed: {result.get('failed', 0)}")

        details = result.get("details", [])
        if details:
            lines.append("  Details:")
            for d in details:
                lines.append(f"    {d}")

        return "\n".join(lines)
