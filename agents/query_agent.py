"""Agent 1: Classifies customer complaints."""

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


QUERY_AGENT_PROMPT = """
You are Kovai Cart's Customer Query Agent.

Analyse the customer's complaint and classify it.

Return these fields:
- Category: Shipping, Refund, Account, Product, Billing, or Other.
- Urgency: Low, Medium, or High.
- Sentiment: Positive, Neutral, Frustrated, or Angry.
- Summary: One concise sentence describing the issue.

Rules:
1. High urgency for threats, fraud, payment disputes, or serious unresolved issues.
2. High urgency when a customer reports receiving a damaged, defective, or broken product and requests a refund or replacement.
3. High urgency when multiple serious order problems occur together, such as delayed delivery and damaged goods.
4. Angry sentiment for aggressive language, threats, or strong dissatisfaction.
5. Medium urgency for delayed orders without additional serious problems.
6. Medium urgency for unresolved service problems that do not meet the High urgency criteria.
7. Low urgency for routine questions and general information requests.
8. Never invent customer or order information.
9. Return a concise, clearly labelled result.
10. Use only the allowed category, urgency and sentiment values.
11. Always return all four fields, even if some information is uncertain.
12. Do not include additional commentary outside the four fields.
13. Base urgency and sentiment on the customer's actual complaint.
14. Do not treat every frustrated customer as Angry.
15. When the complaint involves a damaged product, classify it as Product unless the main issue is specifically the refund process.
16. When the customer requests a refund for a damaged product, do not classify the issue as Medium urgency.

"""


def create_query_agent(
    model_client: OpenAIChatCompletionClient,
) -> AssistantAgent:
    """Create the customer complaint classification agent."""

    return AssistantAgent(
        name="QueryAgent",
        description="Classifies complaints by category, urgency and sentiment.",
        model_client=model_client,
        system_message=QUERY_AGENT_PROMPT,
    )