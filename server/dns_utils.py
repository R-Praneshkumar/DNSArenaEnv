"""DNS zone-file utilities for the OpenEnv RL hackathon.

Pure-Python helpers for rendering, validating, querying, and grading
BIND-style DNS zone files.  **Zero** external dependencies -- stdlib only.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_RTYPES = frozenset(
    {"A", "AAAA", "CNAME", "MX", "NS", "TXT", "SOA", "SRV", "PTR", "CAA"}
)

_MAX_CNAME_HOPS = 10

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DNSRecord:
    """Single DNS resource record."""

    name: str  # e.g., "@", "www", "mail", "_dmarc"
    rtype: str  # A, AAAA, CNAME, MX, NS, TXT, SOA, ...
    rdata: str  # Record data; for MX includes priority like "10 mail.example.com."
    ttl: int | None = None  # None = inherit zone default
    rclass: str = "IN"


# ---------------------------------------------------------------------------
# 1. Rendering
# ---------------------------------------------------------------------------


def render_zone_file(
    records: Sequence[DNSRecord],
    origin: str,
    default_ttl: int = 86400,
) -> str:
    """Render *records* into a BIND-style zone-file string.

    - ``$TTL`` and ``$ORIGIN`` directives at top.
    - SOA formatted with parenthesised fields and comments.
    - Every non-SOA line gets a ``; [i]`` index comment so agents can
      reference records by number.
    """
    origin_dot = _ensure_dot(origin)
    lines: list[str] = [
        f"$TTL {default_ttl}",
        f"$ORIGIN {origin_dot}",
        "",
    ]

    idx = 0
    for rec in records:
        if rec.rtype.upper() == "SOA":
            lines.append(_render_soa(rec, default_ttl))
        else:
            ttl_field = str(rec.ttl) if rec.ttl is not None else ""
            parts = [
                rec.name.ljust(12),
                ttl_field.ljust(8) if ttl_field else "        ",
                rec.rclass,
                rec.rtype,
                rec.rdata,
            ]
            line = "  ".join(parts)
            line = f"{line}  ; [{idx}]"
            lines.append(line)
        idx += 1

    lines.append("")  # trailing newline
    return "\n".join(lines)


def _render_soa(rec: DNSRecord, default_ttl: int) -> str:
    """Format an SOA record with parenthesised fields and comments."""
    ttl_field = str(rec.ttl) if rec.ttl is not None else str(default_ttl)
    # rdata expected format:
    #   "ns1.example.com. admin.example.com. 2024010101 3600 900 604800 86400"
    parts = rec.rdata.split()
    if len(parts) >= 7:
        mname, rname = parts[0], parts[1]
        serial, refresh, retry, expire, minimum = parts[2:7]
    else:
        # Fallback: dump as-is
        return f"{rec.name}  {ttl_field}  {rec.rclass}  SOA  {rec.rdata}"

    soa_lines = [
        f"{rec.name}  {ttl_field}  {rec.rclass}  SOA  {mname} {rname} (",
        f"                        {serial}   ; Serial",
        f"                        {refresh}       ; Refresh",
        f"                        {retry}        ; Retry",
        f"                        {expire}    ; Expire",
        f"                        {minimum} )    ; Minimum / Negative-cache TTL",
    ]
    return "\n".join(soa_lines)


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------


def validate_zone(records: Sequence[DNSRecord], origin: str) -> list[str]:
    """Return a list of human-readable error strings for *records*.

    An empty list means the zone passes all checks.
    """
    errors: list[str] = []
    origin_dot = _ensure_dot(origin)

    # --- record-type check ---
    for i, rec in enumerate(records):
        rt = rec.rtype.upper()
        if rt not in VALID_RTYPES:
            errors.append(
                f"Record [{i}] ({rec.name}): unknown record type '{rec.rtype}'."
            )

    # --- SOA ---
    soa_records = [r for r in records if r.rtype.upper() == "SOA"]
    if len(soa_records) == 0:
        errors.append("Zone must contain exactly one SOA record (none found).")
    elif len(soa_records) > 1:
        errors.append(
            f"Zone must contain exactly one SOA record ({len(soa_records)} found)."
        )

    # --- NS ---
    ns_records = [r for r in records if r.rtype.upper() == "NS"]
    if not ns_records:
        errors.append("Zone must have at least one NS record.")

    # --- per-record checks ---
    name_to_records: Dict[str, list[int]] = {}
    cname_names: set[str] = set()

    for i, rec in enumerate(records):
        rt = rec.rtype.upper()
        norm = _normalise_name(rec.name, origin_dot)
        name_to_records.setdefault(norm, []).append(i)

        if rt == "A":
            _check_ipv4(errors, i, rec)
        elif rt == "AAAA":
            _check_ipv6(errors, i, rec)
        elif rt == "CNAME":
            cname_names.add(norm)
            _check_trailing_dot(errors, i, rec, "CNAME")
        elif rt == "MX":
            _check_mx(errors, i, rec)
        elif rt == "NS":
            _check_trailing_dot(errors, i, rec, "NS")
        elif rt == "TXT":
            _check_txt(errors, i, rec)

    # --- CNAME exclusivity ---
    for cn in cname_names:
        indices = name_to_records.get(cn, [])
        if len(indices) > 1:
            # Check if there are non-CNAME records at that name
            non_cname = [
                j
                for j in indices
                if records[j].rtype.upper() != "CNAME"
            ]
            if non_cname:
                errors.append(
                    f"CNAME exclusivity violation at '{cn}': a CNAME record "
                    f"cannot coexist with other record types (indices {indices})."
                )

    return errors


def _check_ipv4(errors: list[str], idx: int, rec: DNSRecord) -> None:
    rdata = rec.rdata.strip()
    try:
        addr = ipaddress.ip_address(rdata)
        if not isinstance(addr, ipaddress.IPv4Address):
            errors.append(
                f"Record [{idx}] ({rec.name} A): '{rdata}' is not a valid IPv4 address."
            )
    except ValueError:
        errors.append(
            f"Record [{idx}] ({rec.name} A): '{rdata}' is not a valid IPv4 address."
        )


def _check_ipv6(errors: list[str], idx: int, rec: DNSRecord) -> None:
    rdata = rec.rdata.strip()
    try:
        addr = ipaddress.ip_address(rdata)
        if not isinstance(addr, ipaddress.IPv6Address):
            errors.append(
                f"Record [{idx}] ({rec.name} AAAA): '{rdata}' is not a valid IPv6 address."
            )
    except ValueError:
        errors.append(
            f"Record [{idx}] ({rec.name} AAAA): '{rdata}' is not a valid IPv6 address."
        )


def _check_trailing_dot(
    errors: list[str], idx: int, rec: DNSRecord, rtype: str
) -> None:
    """Flag CNAME / NS rdata that looks like a FQDN but lacks a trailing dot."""
    rdata = rec.rdata.strip()
    # For CNAME the rdata is just the target; for NS also just the target
    target = rdata
    if "." in target and not target.endswith("."):
        errors.append(
            f"Record [{idx}] ({rec.name} {rtype}): rdata '{target}' contains dots "
            f"but does not end with '.'; likely missing trailing dot for FQDN."
        )


def _check_mx(errors: list[str], idx: int, rec: DNSRecord) -> None:
    parts = rec.rdata.strip().split(None, 1)
    if len(parts) < 2:
        errors.append(
            f"Record [{idx}] ({rec.name} MX): rdata must be '<priority> <target>' "
            f"but got '{rec.rdata}'."
        )
        return
    pri_str, target = parts
    try:
        int(pri_str)
    except ValueError:
        errors.append(
            f"Record [{idx}] ({rec.name} MX): priority '{pri_str}' is not a valid integer."
        )
    if "." in target and not target.endswith("."):
        errors.append(
            f"Record [{idx}] ({rec.name} MX): target '{target}' contains dots "
            f"but does not end with '.'; likely missing trailing dot."
        )


def _check_txt(errors: list[str], idx: int, rec: DNSRecord) -> None:
    rdata = rec.rdata.strip()
    if not (rdata.startswith('"') and rdata.endswith('"')):
        errors.append(
            f"Record [{idx}] ({rec.name} TXT): rdata should be enclosed in "
            f'double quotes (got: {rdata}).'
        )


# ---------------------------------------------------------------------------
# 3. Dig simulation
# ---------------------------------------------------------------------------


def simulate_dig(
    records: Sequence[DNSRecord],
    origin: str,
    qname: str,
    qtype: str,
) -> str:
    """Simulate a ``dig`` query against *records*.

    Returns human-readable text resembling real ``dig`` output.
    """
    origin_dot = _ensure_dot(origin)
    qtype = qtype.upper()

    # Normalise the query name to a label relative to origin or "@"
    norm_qname = _normalise_name(qname, origin_dot)

    # Build answer section
    answer_lines: list[str] = []
    visited_cnames: set[str] = set()
    current_name: str | None = norm_qname
    hops = 0

    while current_name is not None and hops < _MAX_CNAME_HOPS:
        if current_name in visited_cnames:
            answer_lines.append(f";; CNAME loop detected at {current_name}")
            break
        visited_cnames.add(current_name)

        matches = _find_records(records, origin_dot, current_name, qtype)
        if matches:
            for rec in matches:
                display_name = _display_name(rec.name, origin_dot)
                ttl = rec.ttl if rec.ttl is not None else 86400
                answer_lines.append(
                    f"{display_name}\t{ttl}\t{rec.rclass}\t{rec.rtype}\t{rec.rdata}"
                )
            break
        else:
            # Check for CNAME at current_name (unless qtype is already CNAME)
            if qtype != "CNAME":
                cname_matches = _find_records(
                    records, origin_dot, current_name, "CNAME"
                )
                if cname_matches:
                    cn = cname_matches[0]
                    display_name = _display_name(cn.name, origin_dot)
                    ttl = cn.ttl if cn.ttl is not None else 86400
                    answer_lines.append(
                        f"{display_name}\t{ttl}\t{cn.rclass}\tCNAME\t{cn.rdata}"
                    )
                    # Follow CNAME target
                    target = cn.rdata.strip()
                    current_name = _normalise_name(target, origin_dot)
                    hops += 1
                    continue
            break

    # Authority section: NS records
    ns_lines: list[str] = []
    for rec in records:
        if rec.rtype.upper() == "NS":
            display_name = _display_name(rec.name, origin_dot)
            ttl = rec.ttl if rec.ttl is not None else 86400
            ns_lines.append(
                f"{display_name}\t{ttl}\t{rec.rclass}\tNS\t{rec.rdata}"
            )

    # Format the qname for display in the question section
    display_qname = _display_name(qname, origin_dot)

    output_parts: list[str] = [
        ";; QUESTION SECTION:",
        f";{display_qname}\t\tIN\t{qtype}",
        "",
        ";; ANSWER SECTION:",
    ]
    if answer_lines:
        output_parts.extend(answer_lines)
    else:
        output_parts.append(";; (no records found)")

    output_parts.append("")
    output_parts.append(";; AUTHORITY SECTION:")
    if ns_lines:
        output_parts.extend(ns_lines)
    else:
        output_parts.append(";; (no NS records)")

    output_parts.append("")
    return "\n".join(output_parts)


def _find_records(
    records: Sequence[DNSRecord],
    origin_dot: str,
    norm_qname: str,
    qtype: str,
) -> list[DNSRecord]:
    """Find records matching *norm_qname* and *qtype*."""
    results: list[DNSRecord] = []
    for rec in records:
        rn = _normalise_name(rec.name, origin_dot)
        if rn.lower() != norm_qname.lower():
            continue
        rt = rec.rtype.upper()
        if qtype == "ANY" or rt == qtype:
            results.append(rec)
    return results


def _display_name(name: str, origin_dot: str) -> str:
    """Convert a record name to the FQDN display form used in dig output."""
    n = name.strip().rstrip(".")
    origin_bare = origin_dot.rstrip(".")
    if n == "@" or n == "" or n.lower() == origin_bare.lower():
        return origin_dot
    # If already fully-qualified, just ensure trailing dot
    if n.lower().endswith("." + origin_bare.lower()):
        return _ensure_dot(n)
    # Relative name -> append origin
    return f"{n}.{origin_dot}"


# ---------------------------------------------------------------------------
# 4. Grading
# ---------------------------------------------------------------------------


def grade_zone(
    current_records: Sequence[DNSRecord],
    origin: str,
    required_checks: Sequence[dict],
    original_correct: Sequence[Tuple[str, str, str]] | None = None,
) -> dict:
    """Grade the zone against *required_checks*.

    Weight allocation
    -----------------
    - 30 %  structural validity  (0 validation errors = full marks)
    - 50 %  required resolution checks
    - 20 %  regression (original correct records still present)

    Returns
    -------
    ``{"score": float, "total_weight": float, "passed": int,
       "failed": int, "details": list[str]}``
    """
    origin_dot = _ensure_dot(origin)
    details: list[str] = []

    # ---- 1. Structural validity (30 %) ----
    validation_errors = validate_zone(current_records, origin)
    if not validation_errors:
        validity_score = 1.0
        details.append("Validation: PASS (no errors)")
    else:
        # Each error reduces score; cap at 0
        penalty = min(len(validation_errors) * 0.2, 1.0)
        validity_score = max(1.0 - penalty, 0.0)
        details.append(
            f"Validation: PARTIAL ({len(validation_errors)} error(s), "
            f"score {validity_score:.2f})"
        )
        for e in validation_errors:
            details.append(f"  - {e}")

    # ---- 2. Resolution checks (50 %) ----
    resolution_passed = 0
    resolution_total = 0
    for chk in required_checks:
        if "qname" in chk and "qtype" in chk:
            resolution_total += 1
            ok = _check_resolution(
                current_records, origin_dot, chk
            )
            expected_desc = chk.get("expected_rdata", chk.get("expected_rdata_contains", "?"))
            if ok:
                resolution_passed += 1
                details.append(
                    f"Check {chk['qname']} {chk['qtype']}: PASS"
                )
            else:
                details.append(
                    f"Check {chk['qname']} {chk['qtype']}: FAIL "
                    f"(expected rdata '{expected_desc}')"
                )
        elif chk.get("check") == "no_errors":
            resolution_total += 1
            if not validation_errors:
                resolution_passed += 1
                details.append("Check no_errors: PASS")
            else:
                details.append(
                    f"Check no_errors: FAIL ({len(validation_errors)} error(s))"
                )
        elif chk.get("check") in ("delegation_consistency", "soa_serial_valid"):
            # These are handled as soft checks -- skip scoring for now
            # (they are implicitly captured by the resolution checks above)
            pass

    if resolution_total > 0:
        resolution_score = resolution_passed / resolution_total
    else:
        resolution_score = 1.0

    # ---- 3. Regressions (20 %) ----
    regression_passed = 0
    regression_total = 0
    if original_correct:
        for qname, qtype, expected in original_correct:
            regression_total += 1
            chk = {
                "qname": qname,
                "qtype": qtype,
                "expected_rdata": expected,
            }
            if _check_resolution(current_records, origin_dot, chk):
                regression_passed += 1
            else:
                details.append(
                    f"Regression {qname} {qtype}: FAIL "
                    f"(expected '{expected}' no longer resolves)"
                )
    if regression_total > 0:
        regression_score = regression_passed / regression_total
    else:
        regression_score = 1.0  # nothing to regress

    # ---- Composite ----
    total_weight = 1.0
    score = (
        0.30 * validity_score
        + 0.50 * resolution_score
        + 0.20 * regression_score
    )

    overall_passed = (
        (1 if validity_score == 1.0 else 0)
        + resolution_passed
        + regression_passed
    )
    overall_failed = (
        (0 if validity_score == 1.0 else 1)
        + (resolution_total - resolution_passed)
        + (regression_total - regression_passed)
    )

    return {
        "score": round(score, 4),
        "total_weight": total_weight,
        "passed": overall_passed,
        "failed": overall_failed,
        "details": details,
    }


def _check_resolution(
    records: Sequence[DNSRecord],
    origin_dot: str,
    chk: dict,
) -> bool:
    """Return True if *chk* passes against *records*."""
    qname = chk.get("qname", "@")
    qtype = chk.get("qtype", "A").upper()
    expected = chk.get("expected_rdata", "")
    expected_contains = chk.get("expected_rdata_contains", "")

    norm_qname = _normalise_name(qname, origin_dot)

    # Collect all matching records (following CNAME chain)
    all_matches: list[DNSRecord] = []
    visited: set[str] = set()
    current: str | None = norm_qname
    hops = 0
    while current is not None and hops < _MAX_CNAME_HOPS:
        if current in visited:
            break
        visited.add(current)
        matches = _find_records(records, origin_dot, current, qtype)
        all_matches.extend(matches)
        if matches:
            break
        # Follow CNAME if not already looking for CNAME
        if qtype != "CNAME":
            cname_matches = _find_records(records, origin_dot, current, "CNAME")
            if cname_matches:
                target = cname_matches[0].rdata.strip()
                current = _normalise_name(target, origin_dot)
                hops += 1
                continue
        break

    # Also handle CNAME checks directly
    if qtype == "CNAME" and not all_matches:
        all_matches = _find_records(records, origin_dot, norm_qname, "CNAME")

    # Check exact match
    if expected:
        for m in all_matches:
            if _rdata_match(m.rdata, expected):
                return True

    # Check contains match
    if expected_contains:
        needle = expected_contains.strip().lower()
        for m in all_matches:
            if needle in m.rdata.strip().lower():
                return True

    # If neither expected nor expected_contains, pass if any record exists
    if not expected and not expected_contains:
        return len(all_matches) > 0

    return False


def _rdata_match(actual: str, expected: str) -> bool:
    """Case-insensitive rdata comparison, ignoring trailing dots and whitespace."""
    a = actual.strip().rstrip(".").lower()
    e = expected.strip().rstrip(".").lower()
    return a == e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dot(name: str) -> str:
    """Ensure *name* ends with a single trailing dot."""
    name = name.strip()
    if not name.endswith("."):
        return name + "."
    return name


def _normalise_name(name: str, origin_dot: str) -> str:
    """Normalise a DNS name to a comparable lower-case form relative to zone.

    Examples (origin_dot = "example.com."):
        "@"                 -> "@"
        "example.com."      -> "@"
        "example.com"       -> "@"
        "www.example.com."  -> "www"
        "www"               -> "www"
        "mail.sub"          -> "mail.sub"
    """
    name = name.strip().lower()
    origin_bare = origin_dot.rstrip(".").lower()

    if name in ("@", "", origin_bare, origin_dot.lower()):
        return "@"

    # Strip trailing dot for comparison
    if name.endswith("."):
        name_nodot = name[:-1]
    else:
        name_nodot = name

    if name_nodot == origin_bare:
        return "@"

    suffix = "." + origin_bare
    if name_nodot.endswith(suffix):
        return name_nodot[: -len(suffix)]

    return name_nodot
