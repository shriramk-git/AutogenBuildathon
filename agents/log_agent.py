
"""Agent 4: Records tickets and manages deterministic human escalation."""

import csv
import json
import os
import re
import secrets
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import TICKETS_FILE


# CSV structure required for ticket records.
# Confidence is included to make escalation decisions auditable.
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


def detect_risk_signals(complaint: str) -> tuple[bool, bool]:
    """Detect legal threats and fraud or payment dispute signals."""

    text = complaint.lower()

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
    """
    Apply deterministic rules to decide whether human support is needed.

    High-risk complaints are escalated independently of AI confidence.
    """

    reasons = []

    urgency_value = (urgency or "").strip().lower()
    sentiment_value = (sentiment or "").strip().lower()
    confidence_value = (confidence or "").strip().lower()

    # Escalate urgent or unrecognized classifications.
    if urgency_value == "high":
        reasons.append("High urgency")
    elif urgency_value not in {"low", "medium"}:
        reasons.append("Unrecognized urgency classification")

    # Escalate strongly dissatisfied customers.
    if sentiment_value == "angry":
        reasons.append("Angry customer")

    # Escalate when company policy evidence is unavailable.
    if not knowledge_found:
        reasons.append("No matching knowledge document")

    # Escalate when the proposed answer is not sufficiently supported.
    if confidence_value == "low":
        reasons.append("Low resolution confidence")
    elif confidence_value not in {"high", "medium"}:
        reasons.append("Unrecognized confidence level")

    # Detect serious risks directly from the original complaint.
    # These checks do not depend on the AI classification.
    legal_threat, payment_dispute = detect_risk_signals(complaint)

    if legal_threat:
        reasons.append("Legal threat")

    if payment_dispute:
        reasons.append("Fraud or payment dispute")

    if reasons:
        return "Escalated", "; ".join(reasons)

    return "Resolved", ""


def ensure_csv_schema() -> None:
    """
    Ensure the existing ticket CSV uses the current column structure.

    Older ticket records are preserved. Historical records receive
    'Unknown' for confidence because that information was not recorded
    in the previous CSV format.
    """

    TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not TICKETS_FILE.exists() or TICKETS_FILE.stat().st_size == 0:
        return

    with TICKETS_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        reader = csv.DictReader(csvfile)

        if reader.fieldnames == CSV_COLUMNS:
            return

        existing_rows = list(reader)

    temp_file = TICKETS_FILE.with_name(
        TICKETS_FILE.name + ".tmp"
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
                migrated_row = {
                    column: old_row.get(column, "")
                    for column in CSV_COLUMNS
                }

                if not migrated_row["confidence"]:
                    migrated_row["confidence"] = "Unknown"

                writer.writerow(migrated_row)

        os.replace(temp_file, TICKETS_FILE)

    finally:
        if temp_file.exists():
            temp_file.unlink()


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
    """
    Evaluate escalation and append a complete ticket to the CSV log.

    Args:
        complaint: Original customer complaint.
        category: Classified complaint category.
        urgency: Low, Medium, or High.
        sentiment: Customer sentiment.
        kb_sources: Retrieved knowledge document filenames.
        response: Final customer-facing response.
        confidence: High, Medium, or Low.
        knowledge_found: Whether relevant company knowledge was found.
        customer_name: Customer's name.
        order_id: Customer's order reference.

    Returns:
        JSON containing ticket ID, status and escalation reason.
    """

    TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Normalize confidence to a recognized value.
    confidence_value = (confidence or "").strip().capitalize()

    if confidence_value not in {"High", "Medium", "Low"}:
        confidence_value = "Low"

    # Make the escalation decision using deterministic rules.
    status, escalation_reason = decide_escalation(
        urgency=urgency,
        sentiment=sentiment,
        knowledge_found=knowledge_found,
        confidence=confidence_value,
        complaint=complaint,
    )

    ticket_id = f"KOV-{secrets.token_hex(4).upper()}"

    ticket = {
        "ticket_id": ticket_id,
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "customer_name": customer_name or "Customer",
        "order_id": order_id or "Not provided",
        "complaint": complaint,
        "category": category,
        "urgency": urgency,
        "sentiment": sentiment,
        "kb_sources": "; ".join(kb_sources or []),
        "response": response,
        "confidence": confidence_value,
        "status": status,
        "escalation_reason": escalation_reason,
    }

    # Preserve older ticket records if the CSV schema has changed.
    ensure_csv_schema()

    file_exists = (
        TICKETS_FILE.exists()
        and TICKETS_FILE.stat().st_size > 0
    )

    with TICKETS_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=CSV_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(ticket)

    return json.dumps({
        "ticket_id": ticket_id,
        "status": status,
        "escalation_reason": escalation_reason,
        "confidence": confidence_value,
        "message": "Ticket successfully logged.",
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
