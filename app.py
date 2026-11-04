"""Kovai Cart AI Customer Support Desk - Streamlit application."""

import asyncio
import html
import hmac
import os
import traceback

import pandas as pd
import streamlit as st

from config import TICKETS_FILE
from team import build_ticket_task, create_model_client, create_support_team


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Kovai Cart | Customer Support",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# config.py loads the project's .env before this value is read.
ADMIN_PASSWORD = os.getenv("KOVAI_ADMIN_PASSWORD", "").strip()


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    .hero {
        padding: 20px 24px;
        border-radius: 14px;
        background: linear-gradient(120deg, #142D4E, #245B83);
        color: white;
        margin-bottom: 12px;
    }

    .hero h1 {
        color: white;
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 5px 0;
        padding: 0;
    }

    .hero p {
        color: #E0EAF4;
        font-size: 14px;
        margin: 0;
    }

    div[data-testid="stForm"] {
        border: 1px solid #DCE4EC;
        border-radius: 12px;
        padding: 14px 16px;
    }

    div[data-testid="stFormSubmitButton"] > button,
    div.stButton > button {
        min-height: 42px;
        font-size: 14px;
        font-weight: 700;
        border-radius: 8px;
    }

    div[data-testid="stFormSubmitButton"] button[kind*="primary"] {
        background-color: #1769AA;
        border-color: #1769AA;
        color: white;
    }

    div[data-testid="stFormSubmitButton"] button[kind*="primary"]:hover {
        background-color: #10558D;
        border-color: #10558D;
        color: white;
    }

    div[data-testid="stFormSubmitButton"] button[kind*="secondary"] {
        background-color: #EEF2F6;
        border: 1px solid #CBD5E1;
        color: #263445;
    }

    div[data-testid="stFormSubmitButton"] button[kind*="secondary"]:hover {
        background-color: #DCE4EC;
        border-color: #AAB7C5;
        color: #142D4E;
    }

    .ticket-label {
        font-size: 0.875rem;
        color: #6B7280;
        margin-bottom: 4px;
    }

    .ticket-value {
        font-size: 1.15rem;
        font-weight: 650;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SESSION STATE AND HELPERS
# --------------------------------------------------

def reset_support_form() -> None:
    """Clear form fields and the previous customer result."""
    st.session_state["customer_name_input"] = ""
    st.session_state["order_id_input"] = ""
    st.session_state["complaint_input"] = ""
    st.session_state.pop("latest_ticket", None)
    st.session_state.pop("latest_ticket_was_duplicate", None)


def load_ticket_logs() -> pd.DataFrame:
    """Load ticket history safely; return an empty frame if unavailable."""
    if not TICKETS_FILE.exists() or TICKETS_FILE.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(TICKETS_FILE, keep_default_na=False)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        # Keep technical details in the server console, not in the customer UI.
        print(f"Could not read ticket CSV: {exc}")
        return pd.DataFrame()


def find_ticket(
    updated_logs: pd.DataFrame,
    previous_ticket_ids: set[str],
    complaint: str,
    order_id: str,
) -> tuple[dict | None, bool]:
    """Find a newly created ticket or an existing duplicate ticket.

    Returns (ticket, was_already_present). Matching uses both order ID and
    complaint so a duplicate submission can display its original ticket.
    """
    if updated_logs.empty or "ticket_id" not in updated_logs.columns:
        return None, False

    candidates = updated_logs.copy()

    if "order_id" in candidates.columns:
        candidates = candidates[
            candidates["order_id"].astype(str).str.strip()
            == order_id.strip()
        ]

    if "complaint" in candidates.columns:
        candidates = candidates[
            candidates["complaint"].astype(str).str.strip()
            == complaint.strip()
        ]

    if candidates.empty:
        return None, False

    # The latest matching row is the most relevant record.
    if "timestamp" in candidates.columns:
        candidates = candidates.sort_values(
            "timestamp",
            ascending=False,
            kind="stable",
        )

    ticket = candidates.iloc[0].to_dict()
    was_already_present = (
        str(ticket.get("ticket_id", "")) in previous_ticket_ids
    )
    return ticket, was_already_present


async def execute_support_team(
    complaint: str,
    customer_name: str,
    order_id: str,
) -> None:
    """Run the four internal agents; agent messages stay in the backend."""
    model_client = create_model_client()

    try:
        support_team = create_support_team(model_client)
        task = build_ticket_task(
            complaint=complaint,
            customer_name=customer_name,
            order_id=order_id,
        )

        async for event in support_team.run_stream(task=task):
            source = getattr(event, "source", None)
            content = getattr(event, "content", None)

            # Agent traces are for server-side debugging only.
            if source and content is not None:
                print(f"\n[BACKEND - {source}]")
                print(content)
    finally:
        await model_client.close()


def render_ticket_value(label: str, value: object) -> None:
    """Render a safely escaped ticket summary value."""
    safe_label = html.escape(str(label))
    safe_value = html.escape(
        str(value if value not in (None, "") else "-")
    )
    st.markdown(
        f'<div class="ticket-label">{safe_label}</div>'
        f'<div class="ticket-value">{safe_value}</div>',
        unsafe_allow_html=True,
    )


def filter_ticket_history(
    ticket_df: pd.DataFrame,
    selected_status: str,
    selected_category: str,
    search_text: str,
) -> pd.DataFrame:
    """Apply admin status, category and free-text filters."""
    filtered = ticket_df.copy()

    if selected_status != "All" and "status" in filtered.columns:
        filtered = filtered[
            filtered["status"].astype(str) == selected_status
        ]

    if selected_category != "All" and "category" in filtered.columns:
        filtered = filtered[
            filtered["category"].astype(str) == selected_category
        ]

    query = search_text.strip().casefold()
    if query:
        searchable_columns = [
            column
            for column in (
                "ticket_id",
                "customer_name",
                "order_id",
                "complaint",
                "category",
                "status",
            )
            if column in filtered.columns
        ]

        if searchable_columns:
            matches = pd.Series(False, index=filtered.index)
            for column in searchable_columns:
                matches |= (
                    filtered[column]
                    .astype(str)
                    .str.casefold()
                    .str.contains(query, regex=False, na=False)
                )
            filtered = filtered[matches]

    return filtered


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🛍️ Kovai Cart Support</h1>
        <p>
            We're here to help with your orders, payments,
            returns and account-related questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------

support_tab, admin_tab = st.tabs(
    ["✉️ Contact Support", "🔒 Admin Ticket History"]
)


# --------------------------------------------------
# CUSTOMER SUPPORT TAB
# --------------------------------------------------

with support_tab:
    st.subheader("How can we help you today?")
    st.caption(
        "Describe your issue below. Our support team will review your request."
    )

    with st.form("customer_complaint_form", clear_on_submit=False):
        name_col, order_col = st.columns(2)

        with name_col:
            customer_name = st.text_input(
                "Your name",
                placeholder="Enter your name",
                max_chars=100,
                key="customer_name_input",
            )

        with order_col:
            order_id = st.text_input(
                "Order ID (optional)",
                placeholder="Example: KC10025",
                max_chars=100,
                key="order_id_input",
            )

        complaint = st.text_area(
            "Describe your issue",
            placeholder=(
                "Example: My order was supposed to arrive yesterday, "
                "but I have not received it."
            ),
            height=120,
            max_chars=3000,
            key="complaint_input",
        )

        submit_col, reset_col = st.columns([3, 1])

        with submit_col:
            submitted = st.form_submit_button(
                "Submit Request",
                type="primary",
                use_container_width=True,
            )

        with reset_col:
            st.form_submit_button(
                "↺ Reset",
                type="secondary",
                use_container_width=True,
                on_click=reset_support_form,
            )

    if submitted:
        cleaned_complaint = complaint.strip()
        cleaned_name = customer_name.strip() or "Customer"
        cleaned_order_id = order_id.strip() or "Not provided"

        if not cleaned_complaint:
            st.error("Please describe your issue before submitting.")
        elif len(cleaned_complaint) < 5:
            st.error("Please provide a little more detail about your issue.")
        else:
            st.session_state.pop("latest_ticket", None)
            st.session_state.pop("latest_ticket_was_duplicate", None)

            previous_logs = load_ticket_logs()
            previous_ticket_ids = (
                set(previous_logs["ticket_id"].astype(str))
                if "ticket_id" in previous_logs.columns
                else set()
            )

            try:
                with st.spinner(
                    "We're reviewing your request. Please wait..."
                ):
                    asyncio.run(
                        execute_support_team(
                            complaint=cleaned_complaint,
                            customer_name=cleaned_name,
                            order_id=cleaned_order_id,
                        )
                    )

                updated_logs = load_ticket_logs()
                ticket, was_duplicate = find_ticket(
                    updated_logs=updated_logs,
                    previous_ticket_ids=previous_ticket_ids,
                    complaint=cleaned_complaint,
                    order_id=cleaned_order_id,
                )

                if ticket is None:
                    st.error(
                        "We couldn't confirm that your ticket was recorded. "
                        "Please try again or contact customer support."
                    )
                    print(
                        "ERROR: Workflow completed without a verifiable ticket."
                    )
                else:
                    st.session_state["latest_ticket"] = ticket
                    st.session_state["latest_ticket_was_duplicate"] = (
                        was_duplicate
                    )

                    if was_duplicate:
                        st.info(
                            "We found an existing request with the same "
                            "order ID and complaint. Your original ticket "
                            "reference is shown below."
                        )
                    else:
                        st.success(
                            "Your support request has been recorded successfully."
                        )

            except Exception:
                print("\nBACKEND ERROR:")
                traceback.print_exc()
                st.error(
                    "We're temporarily unable to process your request. "
                    "Please try again later."
                )

    # Customer-facing result: internal agent messages, confidence reasoning
    # and escalation reasons are never displayed in this section.
    ticket = st.session_state.get("latest_ticket")

    if ticket:
        st.divider()
        st.subheader("Your Support Resolution")

        status = str(ticket.get("status", "")).strip()
        response_text = str(ticket.get("response", "")).strip()

        if status == "Resolved":
            st.success("Your request has been resolved.")
        elif status == "Escalated":
            st.warning(
                "Your request has been forwarded to our human support team."
            )
        else:
            st.info("Your support request has been recorded.")

        with st.container(border=True):
            st.write(
                response_text
                or "Your request has been recorded. Our team will review it."
            )

        st.caption("Your ticket reference")
        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            render_ticket_value("Ticket ID", ticket.get("ticket_id", "-"))

        with summary_col2:
            render_ticket_value("Category", ticket.get("category", "-"))

        with summary_col3:
            render_ticket_value("Urgency", ticket.get("urgency", "-"))

        with st.expander("View your ticket details"):
            st.write("**Order ID:**", ticket.get("order_id", "-"))
            st.write("**Status:**", ticket.get("status", "-"))
            st.write("**Submitted complaint:**", ticket.get("complaint", "-"))


# --------------------------------------------------
# ADMIN TICKET HISTORY
# --------------------------------------------------

with admin_tab:
    st.subheader("Admin Ticket History")
    st.caption("Restricted area for authorised support administrators.")

    if not ADMIN_PASSWORD:
        st.warning(
            "Admin access is disabled because KOVAI_ADMIN_PASSWORD "
            "has not been configured. Add it to your local .env file "
            "or your hosting platform's environment variables."
        )
    else:
        if "admin_authenticated" not in st.session_state:
            st.session_state["admin_authenticated"] = False

        if not st.session_state["admin_authenticated"]:
            with st.form("admin_login_form"):
                entered_password = st.text_input(
                    "Admin password",
                    type="password",
                    key="admin_password_input",
                )
                login_submitted = st.form_submit_button(
                    "Log in",
                    type="primary",
                )

            if login_submitted:
                if hmac.compare_digest(entered_password, ADMIN_PASSWORD):
                    st.session_state["admin_authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect admin password.")
        else:
            logout_col, _ = st.columns([1, 4])
            with logout_col:
                if st.button("Log out", key="admin_logout"):
                    st.session_state["admin_authenticated"] = False
                    st.rerun()

            ticket_df = load_ticket_logs()

            if ticket_df.empty:
                st.info("No support tickets have been recorded yet.")
            else:
                status_series = (
                    ticket_df["status"].astype(str)
                    if "status" in ticket_df.columns
                    else pd.Series(
                        ["Unknown"] * len(ticket_df),
                        index=ticket_df.index,
                    )
                )

                total_tickets = len(ticket_df)
                resolved_count = int((status_series == "Resolved").sum())
                escalated_count = int((status_series == "Escalated").sum())

                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Total Tickets", total_tickets)
                metric2.metric("Resolved", resolved_count)
                metric3.metric("Escalated", escalated_count)

                st.divider()
                st.markdown("**Find and filter tickets**")

                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:
                    status_values = (
                        sorted(ticket_df["status"].astype(str).unique())
                        if "status" in ticket_df.columns
                        else []
                    )
                    selected_status = st.selectbox(
                        "Filter by status",
                        ["All"] + status_values,
                    )

                with filter_col2:
                    category_values = (
                        sorted(ticket_df["category"].astype(str).unique())
                        if "category" in ticket_df.columns
                        else []
                    )
                    selected_category = st.selectbox(
                        "Filter by category",
                        ["All"] + category_values,
                    )

                search_text = st.text_input(
                    "Search tickets",
                    placeholder="Search ticket ID, customer, order ID or complaint",
                )

                display_df = filter_ticket_history(
                    ticket_df=ticket_df,
                    selected_status=selected_status,
                    selected_category=selected_category,
                    search_text=search_text,
                )

                st.caption(
                    f"Showing {len(display_df)} of {total_tickets} tickets."
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

                download_col1, download_col2 = st.columns(2)

                with download_col1:
                    filtered_csv = display_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ Download filtered CSV",
                        data=filtered_csv,
                        file_name="kovai_cart_filtered_tickets.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                with download_col2:
                    all_csv = ticket_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ Download all tickets",
                        data=all_csv,
                        file_name="kovai_cart_tickets.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Kovai Cart Customer Support | "
    "AI-powered assistance with human oversight."
)
