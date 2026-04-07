"""DNS debugging tasks for the OpenEnv RL hackathon environment.

Defines three progressively harder DNS debugging scenarios:
  1. fix_single_record  (EASY)   - Fix broken records in a single zone file.
  2. configure_mail     (MEDIUM) - Add missing email infrastructure records.
  3. debug_delegation   (HARD)   - Repair broken three-zone delegation chain.

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
            "emails are not being delivered.\n\n"
            "Known facts:\n"
            "- www.example.com should be a CNAME pointing to example.com\n"
            "- The mail server is at 93.184.216.35\n"
            "- Mail should be delivered via mail.example.com (MX priority 10)\n\n"
            "Examine the zone file and fix the broken records."
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
    """Single-zone task: fix broken email configuration in acme.co.

    The zone already has email records, but they contain five subtle bugs:
      1. "mail" is a CNAME (conflicts with MX) — must be replaced with A record
      2. Both MX records have the same priority (10) — backup should be 20
      3. mail2 has no A record — MX points to it but it can't resolve
      4. SPF has wrong subnet (10.0.2.0/24) and softfail (~all) instead of
         the correct 10.0.1.0/24 with hardfail (-all)
      5. DMARC policy is "none" — should be "quarantine"
    """

    records = [
        DNSRecord("@", "SOA",
                  "ns1.acme.co. hostmaster.acme.co. 2024030101 3600 900 604800 86400"),
        DNSRecord("@", "NS", "ns1.acme.co."),
        DNSRecord("@", "NS", "ns2.acme.co."),
        DNSRecord("@", "A", "10.0.1.1"),
        DNSRecord("www", "CNAME", "acme.co."),
        DNSRecord("ns1", "A", "10.0.1.2"),
        DNSRecord("ns2", "A", "10.0.1.3"),
        # BUG 1: CNAME at "mail" — conflicts with MX pointing to mail.acme.co
        # Agent must DELETE this CNAME and ADD an A record instead
        DNSRecord("mail", "CNAME", "acme.co."),
        # BUG 2: Both MX records have same priority (10) — backup should be 20
        DNSRecord("@", "MX", "10 mail.acme.co."),
        DNSRecord("@", "MX", "10 mail2.acme.co."),
        # BUG 3: mail2 has no A record — MX points to it but it can't resolve
        # (agent must ADD: mail2 A 10.0.1.11)
        # BUG 4: SPF has wrong subnet and softfail instead of hardfail
        DNSRecord("@", "TXT", "\"v=spf1 ip4:10.0.2.0/24 ~all\""),
        # BUG 5: DMARC policy is "none" — should be "quarantine"
        DNSRecord("_dmarc", "TXT",
                  "\"v=DMARC1; p=none; rua=mailto:postmaster@acme.co\""),
    ]

    required_checks = [
        # mail must be an A record (not CNAME) pointing to correct IP
        {"qname": "mail",   "qtype": "A",   "expected_rdata": "10.0.1.10"},
        # mail2 must exist as A record
        {"qname": "mail2",  "qtype": "A",   "expected_rdata": "10.0.1.11"},
        # MX priorities must be correct (10 and 20, not both 10)
        {"qname": "@",      "qtype": "MX",  "expected_rdata": "10 mail.acme.co."},
        {"qname": "@",      "qtype": "MX",  "expected_rdata": "20 mail2.acme.co."},
        # SPF must have correct subnet and hardfail
        {"qname": "@",      "qtype": "TXT", "expected_rdata_contains": "ip4:10.0.1.0/24"},
        {"qname": "@",      "qtype": "TXT", "expected_rdata_contains": "-all"},
        # DMARC must have quarantine policy
        {"qname": "_dmarc", "qtype": "TXT", "expected_rdata_contains": "p=quarantine"},
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
            "The acme.co domain has email delivery problems. Users report "
            "that emails are bouncing, going to spam, and the backup mail "
            "server is unreachable. The email configuration exists but has "
            "multiple issues.\n\n"
            "Known facts:\n"
            "- Primary mail server: mail.acme.co at IP 10.0.1.10 "
            "(MX priority 10)\n"
            "- Backup mail server: mail2.acme.co at IP 10.0.1.11 "
            "(MX priority 20)\n"
            "- SPF should authorize the 10.0.1.0/24 subnet with hard fail "
            "(-all)\n"
            "- DMARC policy should be quarantine, reports to "
            "postmaster@acme.co\n"
            "- The 'mail' hostname must have an A record, not a CNAME "
            "(CNAME conflicts with MX)\n\n"
            "Fix all email configuration issues. Do NOT modify web or "
            "NS records."
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
    """Three-zone task: fix broken delegation chain across grandparent,
    parent, and child zones (corp.com -> infra.corp.com -> db.infra.corp.com).
    """

    # ── Zone 1: corp.com (grandparent) ──────────────────────────────────
    corp_records = [
        DNSRecord("@", "SOA",
                  "ns1.corp.com. admin.corp.com. 2024060101 3600 900 604800 86400"),
        DNSRecord("@", "NS", "ns1.corp.com."),
        DNSRecord("@", "NS", "ns2.corp.com."),
        DNSRecord("@", "A", "10.10.0.1"),
        DNSRecord("www", "CNAME", "corp.com."),
        DNSRecord("ns1", "A", "10.10.0.2"),
        DNSRecord("ns2", "A", "10.10.0.3"),
        # Delegation for infra.corp.com:
        DNSRecord("infra", "NS", "ns1.infra.corp.com."),
        DNSRecord("infra", "NS", "ns2.infra.corp.com."),
        # BUG 1: Stale glue IP — was 10.20.1.10, server migrated but glue
        #         not updated
        DNSRecord("ns1.infra", "A", "10.20.1.99"),
        # BUG 2: Stale glue — wrong IP for ns2 as well
        DNSRecord("ns2.infra", "A", "10.20.1.98"),
        # Red herring: old staging delegation (valid, should NOT be touched)
        DNSRecord("staging", "NS", "ns1.staging.corp.com."),
        DNSRecord("ns1.staging", "A", "10.40.0.5"),
    ]

    # ── Zone 2: infra.corp.com (parent of db) ──────────────────────────
    infra_records = [
        DNSRecord("@", "SOA",
                  "ns1.infra.corp.com. admin.infra.corp.com. 2024060201 3600 900 604800 86400"),
        DNSRecord("@", "NS", "ns1.infra.corp.com."),
        DNSRecord("@", "NS", "ns2.infra.corp.com."),
        DNSRecord("@", "A", "10.20.1.1"),
        DNSRecord("ns1", "A", "10.20.1.10"),
        DNSRecord("ns2", "A", "10.20.1.11"),
        # Red herring: valid SPF record (looks weird but correct — do NOT
        # touch)
        DNSRecord("@", "TXT",
                  "\"v=spf1 ip4:10.20.1.0/24 include:_spf.corp.com -all\""),
        # Delegation for db.infra.corp.com:
        DNSRecord("db", "NS", "ns1.db.infra.corp.com."),
        # BUG 3: wrong NS name — says ns3 but should be ns2
        DNSRecord("db", "NS", "ns3.db.infra.corp.com."),
        # BUG 4: correct glue for ns1 (this one is fine)
        DNSRecord("ns1.db", "A", "10.30.1.10"),
        # BUG 5: glue for wrong hostname (ns3 instead of ns2) AND wrong IP
        DNSRecord("ns3.db", "A", "10.30.1.99"),
    ]

    # ── Zone 3: db.infra.corp.com (child — the broken one) ─────────────
    db_records = [
        # BUG 6: SOA has wrong primary NS (says ns1.infra... should be
        #         ns1.db.infra...)
        DNSRecord("@", "SOA",
                  "ns1.infra.corp.com. admin.db.infra.corp.com. 2024050101 3600 900 604800 86400"),
        DNSRecord("@", "NS", "ns1.db.infra.corp.com."),
        DNSRecord("@", "NS", "ns2.db.infra.corp.com."),
        DNSRecord("ns1", "A", "10.30.1.10"),
        DNSRecord("ns2", "A", "10.30.1.11"),
        # BUG 7: wrong IP for web server — should be 10.30.1.20
        DNSRecord("@", "A", "10.30.1.200"),
        # BUG 8: missing trailing dot on CNAME
        DNSRecord("www", "CNAME", "db.infra.corp.com"),
        # BUG 9: api subdomain points to old decommissioned IP
        DNSRecord("api", "A", "10.30.1.254"),
    ]

    required_checks = [
        # Corp.com delegation glue must be correct
        {"zone": "corp.com", "qname": "ns1.infra", "qtype": "A",
         "expected_rdata": "10.20.1.10"},
        {"zone": "corp.com", "qname": "ns2.infra", "qtype": "A",
         "expected_rdata": "10.20.1.11"},
        # Infra.corp.com delegation must point to correct NS
        {"zone": "infra.corp.com", "qname": "db", "qtype": "NS",
         "expected_rdata": "ns1.db.infra.corp.com."},
        {"zone": "infra.corp.com", "qname": "db", "qtype": "NS",
         "expected_rdata": "ns2.db.infra.corp.com."},
        # Infra.corp.com glue must be correct
        {"zone": "infra.corp.com", "qname": "ns1.db", "qtype": "A",
         "expected_rdata": "10.30.1.10"},
        {"zone": "infra.corp.com", "qname": "ns2.db", "qtype": "A",
         "expected_rdata": "10.30.1.11"},
        # DB zone NS must match delegation
        {"zone": "db.infra.corp.com", "qname": "@", "qtype": "NS",
         "expected_rdata": "ns1.db.infra.corp.com."},
        {"zone": "db.infra.corp.com", "qname": "@", "qtype": "NS",
         "expected_rdata": "ns2.db.infra.corp.com."},
        {"zone": "db.infra.corp.com", "qname": "ns1", "qtype": "A",
         "expected_rdata": "10.30.1.10"},
        {"zone": "db.infra.corp.com", "qname": "ns2", "qtype": "A",
         "expected_rdata": "10.30.1.11"},
        # DB web server must resolve
        {"zone": "db.infra.corp.com", "qname": "@", "qtype": "A",
         "expected_rdata": "10.30.1.20"},
        {"zone": "db.infra.corp.com", "qname": "www", "qtype": "CNAME",
         "expected_rdata": "db.infra.corp.com."},
        # API must point to new server
        {"zone": "db.infra.corp.com", "qname": "api", "qtype": "A",
         "expected_rdata": "10.30.1.30"},
    ]

    original_correct = {
        "corp.com": [
            ("@", "A", "10.10.0.1"),
            ("@", "NS", "ns1.corp.com."),
            ("@", "NS", "ns2.corp.com."),
            ("www", "CNAME", "corp.com."),
            ("ns1", "A", "10.10.0.2"),
            ("ns2", "A", "10.10.0.3"),
            ("staging", "NS", "ns1.staging.corp.com."),
            ("ns1.staging", "A", "10.40.0.5"),
        ],
        "infra.corp.com": [
            ("@", "A", "10.20.1.1"),
            ("@", "NS", "ns1.infra.corp.com."),
            ("@", "NS", "ns2.infra.corp.com."),
            ("ns1", "A", "10.20.1.10"),
            ("ns2", "A", "10.20.1.11"),
        ],
    }

    return {
        "task_id": "debug_delegation",
        "description": (
            "A critical production database at db.infra.corp.com is "
            "unreachable. The DNS delegation chain from corp.com through "
            "infra.corp.com to db.infra.corp.com is broken at multiple "
            "points. You have access to ALL THREE zone files.\n\n"
            "Known facts:\n"
            "- infra.corp.com nameservers: ns1.infra.corp.com (10.20.1.10) "
            "and ns2.infra.corp.com (10.20.1.11)\n"
            "- db.infra.corp.com nameservers: ns1.db.infra.corp.com "
            "(10.30.1.10) and ns2.db.infra.corp.com (10.30.1.11)\n"
            "- db.infra.corp.com web server: 10.30.1.20 "
            "(A record for @ and www)\n"
            "- The corp.com zone was recently migrated and some records "
            "are stale\n"
            "- The api.db.infra.corp.com server was migrated to "
            "10.30.1.30\n"
            "- There are some unrelated records in the zones (staging, "
            "SPF) — do NOT modify those\n"
            "- Do NOT modify any records in the corp.com zone that are "
            "unrelated to the infra delegation\n\n"
            "Fix all delegation issues across the three zones."
        ),
        "zones": {
            "corp.com": corp_records,
            "infra.corp.com": infra_records,
            "db.infra.corp.com": db_records,
        },
        "required_checks": required_checks,
        "original_correct": original_correct,
        "max_steps": 35,
    }
