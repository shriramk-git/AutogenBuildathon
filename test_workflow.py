
"""Automated end-to-end test for Kovai Cart AI Support."""

import asyncio
import csv
import json

from config import TICKETS_FILE
from team import build_ticket_task, create_model_client, create_support_team


async def run_workflow_test():
    print("\n=== KOVAI CART AUTOMATED WORKFLOW TEST ===\n")

    customer_name = "Automated Test Customer"
    order_id = "AUTOTEST001"
    complaint = (
        "My order has not arrived within the estimated delivery "
        "period. Please help me understand what I should do next."
    )

    model_client = create_model_client()
    team = create_support_team(model_client)

    try:
        # Remember the previous ticket count.
        with TICKETS_FILE.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            previous_tickets = list(csv.DictReader(file))

        previous_count = len(previous_tickets)

        # Build and run the four-agent workflow.
        task = build_ticket_task(
            complaint=complaint,
            customer_name=customer_name,
            order_id=order_id,
        )

        print("Submitting test complaint...\n")

        result = await team.run(task=task)

        print("Agent workflow completed.\n")

        # Display the final agent messages.
        for message in result.messages:
            if hasattr(message, "content"):
                content = message.content

                if isinstance(content, str):
                    print(f"{message.source}:\n{content}\n")

        # Verify that exactly one new ticket was recorded.
        with TICKETS_FILE.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            tickets = list(csv.DictReader(file))

        new_tickets = tickets[previous_count:]

        matching_tickets = [
            ticket for ticket in new_tickets
            if ticket.get("order_id") == order_id
        ]

        assert len(matching_tickets) == 1, (
            "Expected exactly one new ticket for AUTOTEST001."
        )

        ticket = matching_tickets[0]

        assert ticket["ticket_id"].startswith("KOV-"), (
            "Ticket ID was not generated correctly."
        )

        assert ticket["category"] == "Shipping", (
            f"Unexpected category: {ticket['category']}"
        )

        assert ticket["kb_sources"].strip(), (
            "No knowledge source was recorded."
        )

        assert ticket["response"].strip(), (
            "Customer response was not recorded."
        )

        assert ticket["status"] in {"Resolved", "Escalated"}, (
            f"Unexpected ticket status: {ticket['status']}"
        )

        print("\n=== AUTOMATED TEST RESULTS ===")
        print("Agent workflow: PASS")
        print("Ticket creation: PASS")
        print("Ticket ID: PASS")
        print("Complaint classification: PASS")
        print("Knowledge source logging: PASS")
        print("Customer response logging: PASS")
        print("Escalation status: PASS")

        print("\nTicket details:")
        print(json.dumps(ticket, indent=4))

        print("\nOVERALL RESULT: PASS")

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(run_workflow_test())
