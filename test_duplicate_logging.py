
"""Safe duplicate ticket protection test."""

import csv
import json
import uuid

from config import TICKETS_FILE
from agents.log_agent import log_ticket


def read_tickets():
    if not TICKETS_FILE.exists():
        return []

    with TICKETS_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def run_test():
    print("\n=== DUPLICATE TICKET TEST ===\n")

    # Unique test data avoids matching previous customer records.
    unique_id = uuid.uuid4().hex[:10]

    customer = "Duplicate Test Customer"
    order_id = f"DUPTEST-{unique_id}"
    complaint = (
        f"Test duplicate protection for reference {unique_id}. "
        "My order delivery status needs checking."
    )

    before = read_tickets()

    try:
        # First submission should create a new ticket.
        first_result = json.loads(
            log_ticket(
                complaint=complaint,
                category="Shipping",
                urgency="Low",
                sentiment="Neutral",
                kb_sources=["shipping_policy.md"],
                response="Please check your tracking link.",
                confidence="High",
                knowledge_found=True,
                customer_name=customer,
                order_id=order_id,
            )
        )

        # Second identical submission should return the same ticket.
        second_result = json.loads(
            log_ticket(
                complaint=complaint,
                category="Shipping",
                urgency="Low",
                sentiment="Neutral",
                kb_sources=["shipping_policy.md"],
                response="Please check your tracking link.",
                confidence="High",
                knowledge_found=True,
                customer_name=customer,
                order_id=order_id,
            )
        )

        after = read_tickets()

        matching = [
            row for row in after
            if row.get("order_id") == order_id
        ]

        # Verify both calls refer to the same ticket.
        assert first_result["ticket_id"] == second_result["ticket_id"], (
            "Duplicate submission created a different ticket ID."
        )

        assert first_result["duplicate"] is False, (
            "First submission should not be marked duplicate."
        )

        assert second_result["duplicate"] is True, (
            "Second submission was not detected as duplicate."
        )

        assert len(matching) == 1, (
            f"Expected one CSV record, found {len(matching)}."
        )

        assert len(after) == len(before) + 1, (
            "CSV should contain exactly one additional ticket."
        )

        print("First submission: PASS")
        print("Second submission detected: PASS")
        print("Same ticket ID returned: PASS")
        print("Only one CSV record created: PASS")

        print("\nTicket ID:", first_result["ticket_id"])
        print("Duplicate flag:", second_result["duplicate"])
        print("\nOVERALL RESULT: PASS")

    finally:
        # Remove only this test's uniquely identified record.
        # Never delete any other ticket.
        current = read_tickets()
        remaining = [
            row for row in current
            if row.get("order_id") != order_id
        ]

        if len(remaining) != len(current):
            temp_file = TICKETS_FILE.with_name(
                TICKETS_FILE.name + ".test.tmp"
            )

            with temp_file.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(current[0].keys()),
                )
                writer.writeheader()
                writer.writerows(remaining)

            temp_file.replace(TICKETS_FILE)

            print("\nTemporary test ticket cleaned up.")


if __name__ == "__main__":
    run_test()
