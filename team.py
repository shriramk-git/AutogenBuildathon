
"""Kovai Cart: Four-agent AutoGen orchestration."""

from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import MODEL_NAME, OPENAI_API_KEY, MAX_MESSAGES
from agents.query_agent import create_query_agent
from agents.knowledge_agent import create_knowledge_agent
from agents.resolution_agent import create_resolution_agent
from agents.log_agent import create_log_agent


def create_model_client() -> OpenAIChatCompletionClient:
    """Create the shared OpenAI model client."""
    return OpenAIChatCompletionClient(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )


def create_support_team(
    model_client: OpenAIChatCompletionClient,
) -> RoundRobinGroupChat:
    """Create the four-agent support team."""

    query_agent = create_query_agent(model_client)
    knowledge_agent = create_knowledge_agent(model_client)
    resolution_agent = create_resolution_agent(model_client)
    log_agent = create_log_agent(model_client)

    termination = (
    TextMentionTermination(
        "LOGGED",
        sources=["LogAgent"],
    )
    | MaxMessageTermination(max_messages=MAX_MESSAGES)
)

    return RoundRobinGroupChat(
        participants=[
            query_agent,
            knowledge_agent,
            resolution_agent,
            log_agent,
        ],
        termination_condition=termination,
    )


def build_ticket_task(
    complaint: str,
    customer_name: str = "Customer",
    order_id: str = "Not provided",
) -> str:
    """Prepare the initial customer support task."""

    return f"""
PROCESS THIS CUSTOMER SUPPORT TICKET FOR KOVAI CART.

CUSTOMER DETAILS
Customer name: {customer_name}
Order ID: {order_id}

CUSTOMER COMPLAINT
{complaint}

WORKFLOW:
1. QueryAgent classifies the complaint.
2. KnowledgeAgent searches the local knowledge base.
3. ResolutionAgent writes a grounded customer response.
4. LogAgent calls log_ticket exactly once.

IMPORTANT:
- Follow the prescribed agent order.
- Do not invent customer information or policies.
- Do not skip any agent.
- Pass the correct information to log_ticket.
- Finish with the exact word LOGGED.
"""
