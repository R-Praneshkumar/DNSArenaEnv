"""DNS debugging tasks for the OpenEnv RL hackathon environment.

Defines three progressively harder DNS debugging scenarios:
  1. fix_single_record  (EASY)   - Fix broken records in a single zone file.
  2. configure_mail     (MEDIUM) - Add missing email infrastructure records.
  3. debug_delegation   (HARD)   - Repair broken subdomain delegation across two zones.

Each task ships a deliberately broken or incomplete set of zone-file records
that an AI agent must diagnose and correct within a step budget.
"""

from __future__ import annotations

try:
    from .dns_utils import DNSRecord
except ImportError:
    from dns_utils import DNSRecord

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASK_IDS: list[str] = [
    "fix_single_record",
    "configure_mail",
    "debug_delegation",
]


def get_task(task_id: str) -> dict:
    """Return the full task configuration for *task_id*.

    Returns
    -------
    dict with keys:
        task_id            – str
        description        – human-readable problem statement
        zones              – dict[str, list[DNSRecord]]  (zone_name -> records)
        required_checks    – list[dict]  grading criteria
        original_correct   – dict[str, list[tuple]]  records that must stay intact
        max_steps          – int  agent step budget
    """
    builders = {
        "fix_single_record": _build_fix_single_record,
        "configure_mail": _build_configure_mail,
        "debug_delegation": _build_debug_delegation,
    }

    builder = builders.get(task_id)
    if builder is None:
        raise ValueError(
            f"Unknown task_id {task_id!r}. "
            f"Valid IDs: {', '.join(TASK_IDS)}"
        )
    return builder()


# ---------------------------------------------------------------------------
# Task 1 — fix_single_record (EASY)
# ---------------------------------------------------------------------------

def _build_fix_single_record() -> dict:
    """Single-zone task: three records have small but critical bugs."""

    records = [
        DNSRecord("@", "SOA",
                  "ns1.example.com. admin.example.com. 2024010101 3600 900 604800 86400"),
        DNSRecord("@", "NS", "ns1.example.com."),
        DNSRecord("@", "NS", "ns2.example.com."),
        DNSRecord("@", "A", "93.184.216.34"),
        # BUG: missing trailing dot — should be "example.com."
        DNSRecord("www", "CNAME", "example.com"),
        # BUG: invalid IP octet (999) — should be "93.184.216.35"
        DNSRecord("mail", "A", "93.184.216.999"),
        # BUG: missing trailing dot — should be "10 mail.example.com."
        DNSRecord("@", "MX", "10 mail.example.com"),
    ]

    required_checks = [
        {"qname": "www",  "qtype": "CNAME", "expected_rdata": "example.com."},
        {"qname": "mail", "qtype": "A",     "expected_rdata": "93.184.216.35"},
        {"qname": "@",    "qtype": "MX",    "expected_rdata": "10 mail.example.com."},
    ]

    original_correct = {
        "example.com": [
            ("@", "A",  "93.184.216.34"),
            ("@", "NS", "ns1.example.com."),
            ("@", "NS", "ns2.example.com."),
        ],
    }

    return {
        "task_id": "fix_single_record",
        "description": (
            "The DNS zone file for example.com has errors causing resolution "
            "failures. The website at www.example.com is unreachable, and "
            "emails are not being delivered. Examine the zone file and fix "
            "the broken records."
        ),
        "zones": {"example.com": records},
        "required_checks": required_checks,
        "original_correct": original_correct,
        "max_steps": 15,
    }


# ---------------------------------------------------------------------------
# Task 2 — configure_mail (MEDIUM)
# ---------------------------------------------------------------------------

def _build_configure_mail() -> dict:
    """Single-zone task: add complete email infrastructure to acme.co."""

    records = [
        DNSRecord("@", "SOA",
                  "ns1.acme.co. hostmaster.acme.co. 2024030101 3600 900 604800 86400"),
        DNSRecord("@", "NS", "ns1.acme.co."),
        DNSRecord("@", "NS", "ns2.acme.co."),
        DNSRecord("@", "A", "10.0.1.1"),
        DNSRecord("www", "CNAME", "acme.co."),
        DNSRecord("ns1", "A", "10.0.1.2"),
        DNSRecord("ns2", "A", "10.0.1.3"),
    ]

    required_checks = [
        {"qname": "mail",   "qtype": "A",   "expected_rdata": "10.0.1.10"},
        {"qname": "mail2",  "qtype": "A",   "expected_rdata": "10.0.1.11"},
        {"qname": "@",      "qtype": "MX",  "expected_rdata_contains": "10 mail.acme.co."},
        {"qname": "@",      "qtype": "MX",  "expected_rdata_contains": "20 mail2.acme.co."},
        {"qname": "@",      "qtype": "TXT", "expected_rdata_contains": "v=spf1"},
        {"qname": "@",      "qtype": "TXT", "expected_rdata_contains": "ip4:10.0.1.0/24"},
        {"qname": "@",      "qtype": "TXT", "expected_rdata_contains": "-all"},
        {"qname": "_dmarc", "qtype": "TXT", "expected_rdata_contains": "v=DMARC1"},
        {"qname": "_dmarc", "qtype": "TXT", "expected_rdata_contains": "p=quarantine"},
        {"qname": "_dmarc", "qtype": "TXT",
         "expected_rdata_contains": "rua=mailto:postmaster@acme.co"},
    ]

    original_correct = {
        "acme.co": [
            ("@",   "A",     "10.0.1.1"),
            ("www", "CNAME", "acme.co."),
            ("@",   "NS",    "ns1.acme.co."),
            ("@",   "NS",    "ns2.acme.co."),
            ("ns1", "A",     "10.0.1.2"),
            ("ns2", "A",     "10.0.1.3"),
        ],
    }

    return {
        "task_id": "configure_mail",
        "description": (
            "The acme.co domain needs complete email delivery configuration. "
            "Currently the zone has basic web records but NO email setup. "
            "Your task: Configure mail delivery according to this "
            "specification:\n\n"
            "- Primary mail server: mail.acme.co at IP 10.0.1.10 "
            "(MX priority 10)\n"
            "- Backup mail server: mail2.acme.co at IP 10.0.1.11 "
            "(MX priority 20)\n"
            "- SPF record: authorize the 10.0.1.0/24 subnet and the mail "
            "servers, hard fail all others\n"
            "- DMARC record: policy=quarantine, send aggregate reports to "
            "postmaster@acme.co\n\n"
            "Do NOT modify existing records."
        ),
        "zones": {"acme.co": records},
        "required_checks": required_checks,
        "original_correct": original_correct,
        "max_steps": 25,
    }


# ---------------------------------------------------------------------------
# Task 3 — debug_delegation (HARD)
# ---------------------------------------------------------------------------

def _build_debug_delegation() -> dict:
    """Two-zone task: fix broken NS delegation between parent and child."""

    parent_records = [
        DNSRecord("@", "SOA",
                  "ns1.parent.org. admin.parent.org. 2024050101 3600 900 604800 86400"),
        DNSRecord("@", "NS", "ns1.parent.org."),
        DNSRecord("@", "NS", "ns2.parent.org."),
        DNSRecord("@", "A", "10.1.0.1"),
        DNSRecord("www", "CNAME", "parent.org."),
        DNSRecord("ns1", "A", "10.1.0.2"),
        DNSRecord("ns2", "A", "10.1.0.3"),
        # Delegation records for dev.parent.org — all three have bugs:
        DNSRecord("dev", "NS", "ns1.dev.parent.org."),
        # BUG: should be ns2, not ns3
        DNSRecord("dev", "NS", "ns3.dev.parent.org."),
        # BUG: wrong glue IP — should be 10.1.1.10
        DNSRecord("ns1.dev", "A", "10.1.1.99"),
        # BUG: wrong glue hostname — should be ns2.dev
        DNSRecord("ns3.dev", "A", "10.1.1.11"),
    ]

    child_records = [
        # BUG: SOA serial (2024040101) is behind parent's (2024050101)
        DNSRecord("@", "SOA",
                  "ns1.dev.parent.org. admin.dev.parent.org. 2024040101 3600 900 604800 86400"),
        DNSRecord("@", "NS", "ns1.dev.parent.org."),
        DNSRecord("@", "NS", "ns2.dev.parent.org."),
        DNSRecord("ns1", "A", "10.1.1.10"),
        DNSRecord("ns2", "A", "10.1.1.11"),
        DNSRecord("@", "A", "10.1.1.20"),
        # BUG: missing trailing dot — should be "dev.parent.org."
        DNSRecord("www", "CNAME", "dev.parent.org"),
    ]

    required_checks = [
        # Parent zone — delegation records
        {"zone": "parent.org", "qname": "dev",      "qtype": "NS",
         "expected_rdata": "ns1.dev.parent.org."},
        {"zone": "parent.org", "qname": "dev",      "qtype": "NS",
         "expected_rdata": "ns2.dev.parent.org."},
        {"zone": "parent.org", "qname": "ns1.dev",  "qtype": "A",
         "expected_rdata": "10.1.1.10"},
        {"zone": "parent.org", "qname": "ns2.dev",  "qtype": "A",
         "expected_rdata": "10.1.1.11"},
        # Child zone — NS and host records
        {"zone": "dev.parent.org", "qname": "@",    "qtype": "NS",
         "expected_rdata": "ns1.dev.parent.org."},
        {"zone": "dev.parent.org", "qname": "@",    "qtype": "NS",
         "expected_rdata": "ns2.dev.parent.org."},
        {"zone": "dev.parent.org", "qname": "ns1",  "qtype": "A",
         "expected_rdata": "10.1.1.10"},
        {"zone": "dev.parent.org", "qname": "ns2",  "qtype": "A",
         "expected_rdata": "10.1.1.11"},
        {"zone": "dev.parent.org", "qname": "@",    "qtype": "A",
         "expected_rdata": "10.1.1.20"},
        {"zone": "dev.parent.org", "qname": "www",  "qtype": "CNAME",
         "expected_rdata": "dev.parent.org."},
        # Cross-zone consistency: parent NS delegation must match child NS
        {"check": "delegation_consistency"},
        # SOA serial in child zone must be valid (>= parent's)
        {"zone": "dev.parent.org", "check": "soa_serial_valid"},
    ]

    parent_original_correct = [
        ("@",   "A",     "10.1.0.1"),
        ("@",   "NS",    "ns1.parent.org."),
        ("@",   "NS",    "ns2.parent.org."),
        ("www", "CNAME", "parent.org."),
        ("ns1", "A",     "10.1.0.2"),
        ("ns2", "A",     "10.1.0.3"),
    ]

    return {
        "task_id": "debug_delegation",
        "description": (
            "The subdomain dev.parent.org is completely unreachable. The "
            "parent zone parent.org delegates to dev.parent.org, but the "
            "delegation is broken. You have access to BOTH zone files. "
            "Debug and fix the delegation so that dev.parent.org resolves "
            "correctly.\n\n"
            "Known facts:\n"
            "- dev.parent.org nameservers should be ns1.dev.parent.org "
            "(10.1.1.10) and ns2.dev.parent.org (10.1.1.11)\n"
            "- dev.parent.org should have a web server at 10.1.1.20 "
            "(A record for @ and www)\n"
            "- The parent zone's NS delegation and glue records must match "
            "the child zone's NS records"
        ),
        "zones": {
            "parent.org": parent_records,
            "dev.parent.org": child_records,
        },
        "required_checks": required_checks,
        "original_correct": {
            "parent.org": parent_original_correct,
        },
        "max_steps": 30,
    }
