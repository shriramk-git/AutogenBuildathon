
"""Agent 2: Retrieves relevant company knowledge using local keyword-based RAG."""

import json
import re
from collections import Counter

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import KNOWLEDGE_BASE_DIR, MAX_KB_RESULTS


# Common words excluded from keyword matching.
STOP_WORDS = {
    "the", "is", "a", "an", "and", "or", "to", "of", "in",
    "for", "on", "with", "my", "i", "me", "it", "this",
    "that", "what", "how", "when", "where", "can", "do",
    "does", "was", "were", "are", "be", "have", "has",
    "had", "you", "your", "we", "our", "please", "from",
    "am", "will", "would", "could", "should", "isnt", "dont",
    "not", "about", "tell", "know", "want", "need"
}


# Lightweight vocabulary expansion for common customer language.
SYNONYMS = {
    "late": ["delay", "delayed", "shipping", "delivery"],
    "delay": ["late", "delayed", "shipping", "delivery"],
    "delayed": ["delay", "late", "shipping", "delivery"],
    "arrive": ["delivery", "shipping", "dispatch"],
    "order": ["shipping", "delivery", "tracking"],
    "refund": ["return", "money", "reimbursement"],
    "return": ["refund", "replacement"],
    "broken": ["damaged", "defective", "replacement"],
    "damaged": ["broken", "defective", "replacement"],
    "defective": ["broken", "damaged", "replacement"],
    "password": ["login", "account", "reset"],
    "login": ["password", "account", "access"],
    "charged": ["payment", "billing", "transaction"],
    "charge": ["payment", "billing", "transaction"],
    "card": ["payment", "billing", "transaction"],
    "multiple": ["duplicate", "charged", "payment"],
    "duplicate": ["multiple", "charged", "payment"],
    "replacement": ["damaged", "broken", "defective"],
    "cancel": ["cancellation", "refund"],
    "cancelled": ["cancellation", "refund"],
    "cancellation": ["cancel", "refund"],
    "tracking": ["shipment", "delivery", "order"],
    "email": ["account", "login", "verification"],
    "locked": ["account", "login", "access"],
}


def tokenize(text: str) -> list[str]:
    """Convert text into searchable lowercase keywords."""

    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def expand_query(tokens: list[str]) -> set[str]:
    """Expand customer keywords with relevant business synonyms."""

    expanded = set(tokens)

    for word in tokens:
        expanded.update(SYNONYMS.get(word, []))

    return expanded


def load_kb_passages() -> list[dict[str, str]]:
    """Load Markdown documents and split them into searchable passages."""

    passages = []

    if not KNOWLEDGE_BASE_DIR.exists():
        return passages

    for file_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            continue

        # Split documents at blank lines.
        chunks = re.split(r"\n\s*\n", content)

        for chunk in chunks:
            cleaned = chunk.strip()

            if cleaned:
                passages.append({
                    "source": file_path.name,
                    "content": cleaned
                })

    return passages


def search_kb(query: str) -> str:
    """
    Search company Markdown documents and return up to three relevant passages.

    Args:
        query: Customer complaint or search query.

    Returns:
        JSON containing relevant passages and source filenames.
    """

    if not query or not query.strip():
        return json.dumps({
            "found": False,
            "message": "Empty search query.",
            "results": []
        })

    passages = load_kb_passages()

    if not passages:
        return json.dumps({
            "found": False,
            "message": "No company knowledge documents found.",
            "results": []
        })

    query_tokens = expand_query(tokenize(query))

    if not query_tokens:
        return json.dumps({
            "found": False,
            "message": "No searchable keywords found.",
            "results": []
        })

    scored_passages = []

    # Score each passage using original customer keywords and synonyms.
    for passage in passages:
        passage_tokens = tokenize(passage["content"])
        token_counts = Counter(passage_tokens)

        matched_terms = query_tokens.intersection(token_counts)

        if not matched_terms:
            continue

        # Give original query terms more importance than expanded synonyms.
        original_tokens = set(tokenize(query))

        original_matches = matched_terms.intersection(original_tokens)
        synonym_matches = matched_terms - original_tokens

        score = (
            len(original_matches) * 2.0
            + len(synonym_matches) * 0.75
        )

        # Slightly reward repeated relevant terms.
        repetition_bonus = sum(
            min(token_counts[word] - 1, 2) * 0.1
            for word in matched_terms
        )

        score += repetition_bonus

        # Reward passages that cover more of the original query.
        coverage = (
            len(original_matches) / len(original_tokens)
            if original_tokens else 0
        )

        score += coverage

        scored_passages.append({
            "source": passage["source"],
            "content": passage["content"],
            "score": round(score, 3),
            "matched_terms": sorted(matched_terms)
        })

    # Highest relevance first, with stable ordering for ties.
    scored_passages.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Require at least one original customer keyword match.
    # Synonym-only matches are not sufficient evidence.
    relevant_passages = [
        item for item in scored_passages
        if set(tokenize(query)).intersection(item["matched_terms"])
    ]

    if not relevant_passages:
        return json.dumps({
            "found": False,
            "message": (
                "No sufficiently relevant information found "
                "in company knowledge documents."
            ),
            "results": []
        })

    # Return no more than the configured maximum.
    result_limit = min(MAX_KB_RESULTS, 3)
    results = relevant_passages[:result_limit]

    return json.dumps({
        "found": True,
        "results": [
            {
                "source": item["source"],
                "passage": item["content"],
                "relevance_score": item["score"]
            }
            for item in results
        ]
    }, indent=2)


KNOWLEDGE_AGENT_PROMPT = """
You are Kovai Cart's Knowledge Base Agent.

YOUR RESPONSIBILITY:
Retrieve accurate company policy information from the local
Markdown knowledge base using the search_kb Python tool.

RULES:
1. Always call the search_kb Python tool.
2. Search using the customer's original complaint and the
   classification summary provided by QueryAgent.
3. Return no more than 3 relevant passages.
4. Include the exact source filename for every retrieved passage.
5. Preserve the meaning of the retrieved passages.
6. Never invent policies, procedures, timelines or company information.
7. Do not use general knowledge as a substitute for company policy.
8. If no sufficiently relevant information is found, explicitly
   report NOT_FOUND.
9. Distinguish between confirmed policy information and missing information.
10. Do not write the customer-facing response.
11. Do not make escalation decisions.
12. Pass the retrieved evidence and source filenames clearly
    to ResolutionAgent.

OUTPUT FORMAT:

KNOWLEDGE_STATUS:
FOUND or NOT_FOUND

RETRIEVED_PASSAGES:
[Relevant passages returned by search_kb, up to 3]

SOURCES:
[Exact Markdown filenames, or None]

If no relevant information is found, do not create or assume
a policy answer.
"""


def create_knowledge_agent(
    model_client: OpenAIChatCompletionClient,
) -> AssistantAgent:
    """Create the knowledge retrieval agent with its Python search tool."""

    return AssistantAgent(
        name="KnowledgeAgent",
        description=(
            "Searches company Markdown documents and retrieves "
            "relevant, source-grounded policy passages."
        ),
        model_client=model_client,
        tools=[search_kb],
        system_message=KNOWLEDGE_AGENT_PROMPT,
        reflect_on_tool_use=True,
    )
