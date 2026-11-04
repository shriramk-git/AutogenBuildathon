"""Agent 4: Records tickets and manages deterministic human escalation.

CSV columns and the log_ticket() tool signature are kept compatible with
the existing Kovai Cart AutoGen 0.7.5 workflow.
"""

import csv
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import TICKETS_FILE


# Keep this schema unchanged so the existing Streamlit UI and tests continue
# to read the same fields.
CSV_COLUMNS = [
    "ticket_id",
    "timestamp",
    "customer_name",
    "order_id",
    "complaint",
    "category",
    "urgency",
    "sentiment",
    "kb_sources",
    "response",
    "confidence",
    "status",
    "escalation_reason",
]

# A repeated submission with the same customer, order ID and complaint within
# this period returns the original ticket instead of creating another row.
DUPLICATE_WINDOW_SECONDS = 300

# Protects migration, duplicate checking and appending inside this Python
# process (including concurrent Streamlit sessions).
_CSV_LOCK = threading.RLock()


def detect_risk_signals(complaint: str) -> tuple[bool, bool]:
    """Detect legal threats and fraud or payment dispute signals."""

    text = (complaint or "").lower()

    legal_patterns = [
        r"\blawyer\b",
        r"\battorney\b",
        r"\bsue\b",
        r"\bcourt\b",
        r"\blegal action\b",
        r"\blegal notice\b",
        r"\bpolice complaint\b",
    ]

    payment_patterns = [
        r"\bfraud\b",
        r"\bunauthori[sz]ed\b",
        r"\bcharged my card\b",
        r"\bduplicate charge\b",
        r"\bcharged\b.*\btwice\b",
        r"\bcharged\b.*\bthree times\b",
        r"\bcharged twice\b",
        r"\bcharged three times\b",
        r"\bchargeback\b",
        r"\bcontacting my bank\b",
        r"\bpayment dispute\b",
        r"\bstolen card\b",
        r"\btransaction I did not make\b",
        r"\btransaction I didn't make\b",
    ]

    legal_threat = any(
        re.search(pattern, text) for pattern in legal_patterns
    )
    payment_dispute = any(
        re.search(pattern, text) for pattern in payment_patterns
    )

    return legal_threat, payment_dispute


def decide_escalation(
    urgency: str,
    sentiment: str,
    knowledge_found: bool,
    confidence: str,
    complaint: str,
) -> tuple[str, str]:
    """Apply the existing deterministic human-escalation rules."""

    reasons = []

    urgency_value = (urgency or "").strip().lower()
    sentiment_value = (sentiment or "").strip().lower()
    confidence_value = (confidence or "").strip().lower()

    # High urgency and unknown urgency classifications are escalated.
    if urgency_value == "high":
        reasons.append("High urgency")
    elif urgency_value not in {"low", "medium"}:
        reasons.append("Unrecognized urgency classification")

    # Escalate strongly dissatisfied customers.
    if sentiment_value == "angry":
        reasons.append("Angry customer")

    # Escalate when relevant company knowledge is unavailable.
    if not knowledge_found:
        reasons.append("No matching knowledge document")

    # Escalate when the proposed answer is not sufficiently supported.
    if confidence_value == "low":
        reasons.append("Low resolution confidence")
    elif confidence_value not in {"high", "medium"}:
        reasons.append("Unrecognized confidence level")

    # Detect risk directly from the original complaint.
    legal_threat, payment_dispute = detect_risk_signals(complaint)

    if legal_threat:
        reasons.append("Legal threat")

    if payment_dispute:
        reasons.append("Fraud or payment dispute")

    if reasons:
        return "Escalated", "; ".join(reasons)

    return "Resolved", ""


def _normalize_for_duplicate_check(value: str | None) -> str:
    """Normalize text so trivial whitespace/case differences still match."""

    return " ".join((value or "").strip().casefold().split())


def _duplicate_fingerprint(
    customer_name: str,
    order_id: str,
    complaint: str,
) -> str:
    """Create a stable fingerprint without storing extra CSV columns."""

    parts = (
        _normalize_for_duplicate_check(customer_name or "Customer"),
        _normalize_for_duplicate_check(order_id or "Not provided"),
        _normalize_for_duplicate_check(complaint),
    )
    raw_value = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(raw_value).hexdigest()


def _parse_timestamp(value: str) -> datetime | None:
    """Parse a stored ISO timestamp into an aware datetime."""

    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        return parsed
    except (TypeError, ValueError):
        return None


def _find_recent_duplicate(
    rows: list[dict[str, str]],
    customer_name: str,
    order_id: str,
    complaint: str,
) -> dict[str, str] | None:
    """Return a matching recent ticket, if one exists."""

    target_fingerprint = _duplicate_fingerprint(
        customer_name,
        order_id,
        complaint,
    )
    now = datetime.now().astimezone()
    cutoff = now - timedelta(seconds=DUPLICATE_WINDOW_SECONDS)

    # Search newest records first.
    for row in reversed(rows):
        existing_fingerprint = _duplicate_fingerprint(
            row.get("customer_name", ""),
            row.get("order_id", ""),
            row.get("complaint", ""),
        )

        if existing_fingerprint != target_fingerprint:
            continue

        created_at = _parse_timestamp(row.get("timestamp", ""))
        if created_at is None:
            continue

        if cutoff <= created_at <= now + timedelta(seconds=5):
            return row

    return None


def ensure_csv_schema() -> None:
    """Create or migrate the CSV while preserving existing ticket records."""

    with _CSV_LOCK:
        TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)

        if not TICKETS_FILE.exists() or TICKETS_FILE.stat().st_size == 0:
            return

        try:
            with TICKETS_FILE.open(
                mode="r",
                newline="",
                encoding="utf-8-sig",
            ) as csvfile:
                reader = csv.DictReader(csvfile)
                existing_headers = reader.fieldnames or []
                existing_rows = list(reader)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise RuntimeError(
                f"Could not read ticket CSV safely: {exc}"
            ) from exc

        if existing_headers == CSV_COLUMNS:
            return

        # A non-empty file without a usable header is unsafe to migrate
        # automatically. Stop rather than silently losing customer records.
        if not existing_headers:
            raise RuntimeError(
                "Ticket CSV has no readable header. Back it up and inspect "
                "the file before logging new tickets."
            )

        temp_file = TICKETS_FILE.with_name(
            f"{TICKETS_FILE.name}.tmp"
        )

        try:
            with temp_file.open(
                mode="w",
                newline="",
                encoding="utf-8",
            ) as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=CSV_COLUMNS,
                    extrasaction="ignore",
                )
                writer.writeheader()

                for old_row in existing_rows:
                    # Ignore DictReader's overflow field if a malformed row
                    # contains more values than the header.
                    if None in old_row:
                        old_row.pop(None, None)

                    migrated_row = {
                        column: old_row.get(column, "") or ""
                        for column in CSV_COLUMNS
                    }

                    if not migrated_row["confidence"]:
                        migrated_row["confidence"] = "Unknown"

                    writer.writerow(migrated_row)

                csvfile.flush()
                os.fsync(csvfile.fileno())

            # Atomic replacement prevents readers from seeing a half-written
            # migrated CSV.
            os.replace(temp_file, TICKETS_FILE)

        except OSError as exc:
            raise RuntimeError(
                f"Could not safely migrate ticket CSV: {exc}"
            ) from exc
        finally:
            if temp_file.exists():
                temp_file.unlink()


def _read_ticket_rows() -> list[dict[str, str]]:
    """Read the current CSV while the caller holds the CSV lock."""

    if not TICKETS_FILE.exists() or TICKETS_FILE.stat().st_size == 0:
        return []

    try:
        with TICKETS_FILE.open(
            mode="r",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            return list(csv.DictReader(csvfile))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise RuntimeError(
            f"Could not read ticket records: {exc}"
        ) from exc


def log_ticket(
    complaint: str,
    category: str,
    urgency: str,
    sentiment: str,
    kb_sources: list[str],
    response: str,
    confidence: str,
    knowledge_found: bool,
    customer_name: str = "Customer",
    order_id: str = "Not provided",
) -> str:
    """Evaluate escalation and append a complete ticket to the CSV log.

    The public signature is intentionally unchanged for AutoGen tool use.

    Repeated submissions with the same customer, order ID and complaint
    within DUPLICATE_WINDOW_SECONDS return the existing ticket instead of
    appending a second record.
    """

    if not isinstance(complaint, str) or not complaint.strip():
        raise ValueError("A non-empty customer complaint is required.")

    customer_name = (customer_name or "Customer").strip() or "Customer"
    order_id = (order_id or "Not provided").strip() or "Not provided"

    # Normalize confidence to a recognized value.
    confidence_value = (confidence or "").strip().capitalize()
    if confidence_value not in {"High", "Medium", "Low"}:
        confidence_value = "Low"

    # Preserve the existing deterministic escalation rules.
    status, escalation_reason = decide_escalation(
        urgency=urgency,
        sentiment=sentiment,
        knowledge_found=knowledge_found,
        confidence=confidence_value,
        complaint=complaint,
    )

    ticket_id = f"KOV-{os.urandom(4).hex().upper()}"

    ticket = {
        "ticket_id": ticket_id,
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "customer_name": customer_name,
        "order_id": order_id,
        "complaint": complaint.strip(),
        "category": (category or "Other").strip(),
        "urgency": (urgency or "Unknown").strip(),
        "sentiment": (sentiment or "Unknown").strip(),
        "kb_sources": "; ".join(
            source.strip()
            for source in (kb_sources or [])
            if isinstance(source, str) and source.strip()
        ),
        "response": (response or "").strip(),
        "confidence": confidence_value,
        "status": status,
        "escalation_reason": escalation_reason,
    }

    with _CSV_LOCK:
        TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Migration and duplicate detection must happen under the same lock
        # as the append, so concurrent threads cannot both pass the check.
        ensure_csv_schema()
        existing_rows = _read_ticket_rows()

        duplicate = _find_recent_duplicate(
            rows=existing_rows,
            customer_name=customer_name,
            order_id=order_id,
            complaint=complaint,
        )

        if duplicate:
            return json.dumps({
                "ticket_id": duplicate.get("ticket_id", ""),
                "status": duplicate.get("status", ""),
                "escalation_reason": duplicate.get(
                    "escalation_reason", ""
                ),
                "confidence": duplicate.get("confidence", "Unknown"),
                "message": (
                    "Duplicate submission detected. "
                    "The existing ticket was retained."
                ),
                "duplicate": True,
            })

        file_exists = (
            TICKETS_FILE.exists()
            and TICKETS_FILE.stat().st_size > 0
        )

        try:
            with TICKETS_FILE.open(
                mode="a",
                newline="",
                encoding="utf-8",
            ) as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=CSV_COLUMNS,
                    extrasaction="ignore",
                )

                if not file_exists:
                    writer.writeheader()

                writer.writerow(ticket)
                csvfile.flush()
                os.fsync(csvfile.fileno())

        except (OSError, csv.Error) as exc:
            raise RuntimeError(
                f"Could not write the support ticket: {exc}"
            ) from exc

    return json.dumps({
        "ticket_id": ticket_id,
        "status": status,
        "escalation_reason": escalation_reason,
        "confidence": confidence_value,
        "message": "Ticket successfully logged.",
        "duplicate": False,
    })


LOG_AGENT_PROMPT = """
You are Kovai Cart's Ticket Logging Agent.

YOUR RESPONSIBILITY:
Collect the completed support information, call log_ticket exactly
once, and report the recorded ticket result.

INFORMATION TO COLLECT:

1. QueryAgent:
   - Category
   - Urgency
   - Sentiment
   - Original customer complaint

2. KnowledgeAgent:
   - Whether relevant passages were found
   - Source filenames

3. ResolutionAgent:
   - Complete customer-facing reply
   - Confidence level

TOOL USAGE RULES:

1. Call log_ticket exactly once for the current customer ticket.
2. Pass the original complaint without changing its meaning.
3. Pass the classification values from QueryAgent.
4. Pass source filenames as a list of strings.
5. Pass knowledge_found as true only when relevant passages
   were actually retrieved.
6. Pass the confidence level from ResolutionAgent.
7. Do not modify or rewrite the customer response.
8. Do not decide escalation yourself. The log_ticket tool applies
   the deterministic escalation rules.
9. Do not invent missing customer or order information.
10. Do not claim that a ticket was logged unless the tool succeeds.
11. If the tool reports duplicate=true, report the existing ticket ID.
    Do not call the tool again.

FINAL RESPONSE:

After successful tool execution, report:
- Ticket ID
- Final status (Resolved or Escalated)
- Escalation reason, if present

Do not expose internal agent analysis to the customer.

End your final message with the exact word LOGGED.
"""


def create_log_agent(
    model_client: OpenAIChatCompletionClient,
) -> AssistantAgent:
    """Create the final ticket logging and escalation agent."""

    return AssistantAgent(
        name="LogAgent",
        description=(
            "Records support tickets, applies deterministic "
            "escalation rules and reports ticket status."
        ),
        model_client=model_client,
        tools=[log_ticket],
        system_message=LOG_AGENT_PROMPT,
        reflect_on_tool_use=True,
    )
