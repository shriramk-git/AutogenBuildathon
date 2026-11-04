
"""End-to-end workflow tests for Kovai Cart AI Support."""

import asyncio
import csv
import json
from datetime import datetime

from config import TICKETS_FILE
from team import build_ticket_task, create_model_client, create_support_team


TEST_CASES = [
    {
        "name": "Normal delayed delivery",
        "complaint": (
            "My order has not arrived within the estimated delivery "
            "period. Please help me understand what I should do next."
        ),
        "category": "Shipping",
        "expected_status": "Resolved",
    },
    {
        "name": "Damaged product and refund request",
        "complaint": (
            "My blender arrived broken. I want a refund immediately. "
            "This is very poor service."
        ),
        "category": "Product",
        "expected_status": "Escalated",
    },
    {
        "name": "Unauthorised payment",
        "complaint": (
            "There is an unauthorised transaction on my card. "
            "I did not make this payment and want it investigated."
        ),
        "category": None,
        "expected_status": "Escalated",
    },
    {
        "name": "Legal threat",
        "complaint": (
            "If you do not resolve my complaint, I will take legal action."
        ),
        "category": None,
        "expected_status": "Escalated",
    },
    {
        "name": "Question outside the knowledge base",
        "complaint": (
            "What colour shirt was worn by the warehouse manager yesterday?"
        ),
        "category": None,
        "expected_status": "Escalated",
    },
]


def read_tickets():
    """Read existing ticket records safely."""
    if not TICKETS_FILE.exists():
        return []

    with TICKETS_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


async def run_workflow_test():
    print("\n=== KOVAI CART WORKFLOW TEST SUITE ===\n")

    model_client = create_model_client()
    results = []

    # Unique IDs prevent previous test runs from affecting assertions.
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")

    try:
        for index, test_case in enumerate(TEST_CASES, start=1):
            order_id = f"AUTOTEST{run_id}{index:02d}"

            print(f"\nTEST {index}: {test_case['name']}")
            print("-" * 55)

            previous_ids = {
                ticket.get("ticket_id")
                for ticket in read_tickets()
            }

            team = create_support_team(model_client)

            task = build_ticket_task(
                complaint=test_case["complaint"],
                customer_name="Automated Test Customer",
                order_id=order_id,
            )

            result = await team.run(task=task)

            # Find the ticket created by this workflow.
            new_tickets = [
                ticket
                for ticket in read_tickets()
                if ticket.get("ticket_id") not in previous_ids
                and ticket.get("order_id") == order_id
            ]

            assert len(new_tickets) == 1, (
                f"Expected exactly one ticket for {order_id}; "
                f"found {len(new_tickets)}."
            )

            ticket = new_tickets[0]

            # Verify ticket logging.
            assert ticket["ticket_id"].startswith("KOV-")
            assert ticket["response"].strip()
            assert ticket["status"] in {"Resolved", "Escalated"}

            # Verify classification when an exact category is expected.
            if test_case["category"]:
                assert ticket["category"] == test_case["category"], (
                    f"Expected category {test_case['category']}, "
                    f"got {ticket['category']}."
                )

            # Verify the expected escalation decision.
            assert ticket["status"] == test_case["expected_status"], (
                f"Expected {test_case['expected_status']}, "
                f"got {ticket['status']}."
            )

            # Normal answers should have supporting knowledge sources.
            if test_case["name"] == "Normal delayed delivery":
                assert ticket["kb_sources"].strip(), (
                    "Knowledge sources were not recorded."
                )

            results.append({
                "test": test_case["name"],
                "status": "PASS",
                "ticket_id": ticket["ticket_id"],
                "actual_status": ticket["status"],
            })

            print("PASS")
            print(f"Ticket ID: {ticket['ticket_id']}")
            print(f"Category: {ticket['category']}")
            print(f"Status: {ticket['status']}")

        print("\n=== FINAL TEST RESULTS ===\n")
        print(json.dumps(results, indent=4))

        print(
            f"\nRESULT: {len(results)}/{len(TEST_CASES)} TESTS PASSED"
        )
        print("\nOVERALL RESULT: PASS")

    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(run_workflow_test())
