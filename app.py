"""Kovai Cart AI Customer Support Desk."""

import asyncio
import html
import traceback

import pandas as pd
import streamlit as st

from config import TICKETS_FILE
from team import (
    create_model_client,
    create_support_team,
    build_ticket_task,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Kovai Cart | Customer Support",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# COMPACT CUSTOMER-FACING DESIGN
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

    /* Use more of the available screen and reduce vertical whitespace. */
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

    /* Keep buttons compact and visually distinct. */
    div[data-testid="stFormSubmitButton"] > button,
    div.stButton > button {
        min-height: 42px;
        font-size: 14px;
        font-weight: 700;
        border-radius: 8px;
    }

    /* Primary Submit button. */
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

    /* Neutral Reset button. */
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

    h2 {
        font-weight: 750;
    }

    .ticket-id-label {
        font-size: 0.875rem;
        color: #6B7280;
        margin-bottom: 4px;
    }

    .ticket-id-value {
        font-size: 1.25rem;
        font-weight: 600;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    /* Reduce excess spacing between Streamlit elements. */
    div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# RESET FORM CALLBACK
# --------------------------------------------------

def reset_support_form():
    """Clear the support form and previous ticket result."""

    st.session_state["customer_name_input"] = ""
    st.session_state["order_id_input"] = ""
    st.session_state["complaint_input"] = ""
    st.session_state.pop("latest_ticket", None)


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
# LOAD TICKET HISTORY
# --------------------------------------------------

def load_ticket_logs() -> pd.DataFrame:
    """Load customer support ticket history from CSV."""

    if not TICKETS_FILE.exists():
        return pd.DataFrame()

    if TICKETS_FILE.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(TICKETS_FILE).fillna("")

    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


# --------------------------------------------------
# BACKEND: RUN THE FOUR INTERNAL AGENTS
# --------------------------------------------------

async def execute_support_team(
    complaint: str,
    customer_name: str,
    order_id: str,
) -> None:
    """
    Execute the internal AutoGen support team.

    Agent activity is printed to the backend terminal
    and is never displayed in the customer interface.
    """

    model_client = create_model_client()

    try:
        team = create_support_team(model_client)

        task = build_ticket_task(
            complaint=complaint,
            customer_name=customer_name,
            order_id=order_id,
        )

        async for event in team.run_stream(task=task):
            source = getattr(event, "source", None)
            content = getattr(event, "content", None)

            if source and content is not None:
                print(f"\n[BACKEND - {source}]")
                print(content)

    finally:
        await model_client.close()


# --------------------------------------------------
# CUSTOMER SUPPORT INTERFACE
# --------------------------------------------------

submit_tab, logs_tab = st.tabs(
    [
        "✉️ Contact Support",
        "📋 Ticket History (Admin)",
    ]
)


# --------------------------------------------------
# TAB 1: SUBMIT COMPLAINT
# --------------------------------------------------

with submit_tab:

    st.subheader("How can we help you today?")

    st.caption(
        "Describe your issue below. "
        "Our support team will review your request."
    )

    with st.form("customer_complaint_form"):

        # Name and Order ID appear on the same row.
        name_col, order_col = st.columns(2)

        with name_col:
            customer_name = st.text_input(
                "Your name",
                placeholder="Enter your name",
                key="customer_name_input",
            )

        with order_col:
            order_id = st.text_input(
                "Order ID (optional)",
                placeholder="Example: KC10025",
                key="order_id_input",
            )

        complaint = st.text_area(
            "Describe your issue",
            placeholder=(
                "Example: My order was supposed to arrive "
                "yesterday, but I have not received it."
            ),
            height=120,
            key="complaint_input",
        )

        # Both actions stay on the same row.
        # Submit is wide and primary; Reset is compact and neutral.
        submit_col, reset_col = st.columns([3, 1])

        with submit_col:
            submitted = st.form_submit_button(
                "Submit",
                type="primary",
                use_container_width=True,
            )

        with reset_col:
            reset_clicked = st.form_submit_button(
                "↺ Reset",
                type="secondary",
                use_container_width=True,
                on_click=reset_support_form,
            )

    # --------------------------------------------------
    # PROCESS SUBMITTED COMPLAINT
    # --------------------------------------------------

    if submitted:

        if not complaint.strip():

            st.error(
                "Please describe your issue before submitting."
            )

        else:

            st.session_state.pop("latest_ticket", None)

            previous_logs = load_ticket_logs()
            previous_count = len(previous_logs)

            try:

                with st.spinner(
                    "We're reviewing your request. Please wait..."
                ):

                    asyncio.run(
                        execute_support_team(
                            complaint=complaint.strip(),
                            customer_name=(
                                customer_name.strip() or "Customer"
                            ),
                            order_id=(
                                order_id.strip() or "Not provided"
                            ),
                        )
                    )

                updated_logs = load_ticket_logs()

                # Confirm that a new ticket was actually recorded.
                if len(updated_logs) <= previous_count:

                    st.error(
                        "We couldn't complete your request right now. "
                        "Please try again or contact customer support."
                    )

                    print(
                        "ERROR: Workflow finished without recording a ticket."
                    )

                else:

                    latest_ticket = updated_logs.iloc[-1].to_dict()

                    st.session_state["latest_ticket"] = latest_ticket

                    st.success(
                        "Your support request has been processed successfully!"
                    )

            except Exception:

                print("\nBACKEND ERROR:")
                traceback.print_exc()

                st.error(
                    "We're temporarily unable to process your request. "
                    "Please try again later."
                )

    # --------------------------------------------------
    # DISPLAY CUSTOMER RESOLUTION
    # --------------------------------------------------

    ticket = st.session_state.get("latest_ticket")

    if ticket:

        st.divider()

        st.subheader("Your Support Resolution")

        status = str(
            ticket.get("status", "")
        ).strip()

        if status == "Resolved":

            st.success("✅ Your request has been resolved.")

            with st.container(border=True):

                st.subheader(
                    "Thank you for contacting Kovai Cart!"
                )

                st.write(
                    ticket.get("response", "")
                )

        else:

            st.warning(
                "Your request has been forwarded to our human support team."
            )

            with st.container(border=True):

                st.subheader(
                    "Additional assistance is required."
                )

                st.write(
                    ticket.get("response", "")
                )

                escalation_reason = ticket.get(
                    "escalation_reason",
                    "Additional review required",
                )

                st.warning(
                "Your request has been forwarded to our support team "
                "for further assistance."
                )

        # --------------------------------------------------
        # TICKET SUMMARY
        # --------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                '<div class="ticket-id-label">Ticket ID</div>',
                unsafe_allow_html=True,
            )

            ticket_id_html = html.escape(
                str(ticket.get("ticket_id", "-"))
            )

            st.markdown(
                f'<div class="ticket-id-value">{ticket_id_html}</div>',
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                '<div class="ticket-id-label">Category</div>',
                unsafe_allow_html=True,
            )

            category_html = html.escape(
                str(ticket.get("category", "-"))
            )

            st.markdown(
                f'<div class="ticket-id-value">{category_html}</div>',
                unsafe_allow_html=True,
            )

        with col3:

            st.markdown(
                '<div class="ticket-id-label">Urgency</div>',
                unsafe_allow_html=True,
            )

            urgency_html = html.escape(
                str(ticket.get("urgency", "-"))
            )

            st.markdown(
                f'<div class="ticket-id-value">{urgency_html}</div>',
                unsafe_allow_html=True,
            )

        # --------------------------------------------------
        # TICKET DETAILS
        # --------------------------------------------------

        with st.expander("View ticket details"):

            st.write(
                "**Customer:**",
                ticket.get("customer_name"),
            )

            st.write(
                "**Order ID:**",
                ticket.get("order_id"),
            )

            st.write(
                "**Category:**",
                ticket.get("category"),
            )

            st.write(
                "**Urgency:**",
                ticket.get("urgency"),
            )

            st.write(
                "**Ticket ID:**",
                ticket.get("ticket_id"),
            )

            st.write(
                "**Status:**",
                ticket.get("status"),
            )


# --------------------------------------------------
# TAB 2: TICKET HISTORY (ADMIN)
# --------------------------------------------------

with logs_tab:

    st.subheader("Ticket History (Admin)")

    ticket_df = load_ticket_logs()

    if ticket_df.empty:

        st.info(
            "No support tickets have been recorded yet."
        )

    else:

        total_tickets = len(ticket_df)

        resolved_count = (
            ticket_df["status"] == "Resolved"
        ).sum()

        escalated_count = (
            ticket_df["status"] == "Escalated"
        ).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Tickets",
            total_tickets,
        )

        col2.metric(
            "Resolved",
            int(resolved_count),
        )

        col3.metric(
            "Escalated",
            int(escalated_count),
        )

        st.divider()

        status_filter = st.selectbox(
            "Filter tickets by status",
            [
                "All",
                "Resolved",
                "Escalated",
            ],
        )

        display_df = ticket_df.copy()

        if status_filter != "All":

            display_df = display_df[
                display_df["status"] == status_filter
            ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = ticket_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Ticket History (CSV)",
            data=csv_data,
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
