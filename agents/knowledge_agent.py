"""Agent 2: Retrieves policy evidence from the local Markdown knowledge base.

This module uses lightweight, section-aware keyword retrieval. It does not
require embeddings, a vector database, or additional packages.
"""

import json
import re
from collections import Counter
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import KNOWLEDGE_BASE_DIR, MAX_KB_RESULTS


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "can", "could", "did", "do", "does", "for", "from", "had",
    "has", "have", "how", "i", "if", "in", "is", "it", "me",
    "my", "of", "on", "or", "our", "please", "should", "so",
    "that", "the", "their", "them", "there", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your",
}

# Small, explicit vocabulary expansion for common e-commerce language.
SYNONYMS = {
    "late": ["delay", "delayed", "delivery", "shipment"],
    "delay": ["late", "delayed", "delivery", "shipment"],
    "delayed": ["delay", "late", "delivery", "shipment"],
    "arrive": ["delivery", "shipping", "dispatch"],
    "arrived": ["delivery", "shipping", "dispatch"],
    "order": ["shipment", "delivery", "tracking"],
    "package": ["parcel", "order", "shipment"],
    "parcel": ["package", "order", "shipment"],
    "shipment": ["package", "parcel", "delivery"],
    "refund": ["reimbursement", "money back", "return"],
    "money": ["refund", "payment", "reimbursement"],
    "return": ["refund", "replacement"],
    "broken": ["damaged", "defective"],
    "damage": ["damaged", "defective", "broken"],
    "damaged": ["damage", "defective", "broken"],
    "defective": ["broken", "damaged", "faulty"],
    "faulty": ["defective", "broken", "damaged"],
    "exchange": ["replacement", "return"],
    "replace": ["replacement", "exchange"],
    "replacement": ["replace", "exchange", "damaged"],
    "password": ["login", "account", "reset"],
    "forgot": ["forgotten", "reset", "password"],
    "login": ["password", "account", "access"],
    "locked": ["account", "login", "access"],
    "charged": ["payment", "billing", "transaction"],
    "charge": ["payment", "billing", "transaction"],
    "card": ["payment", "billing", "transaction"],
    "duplicate": ["multiple", "charged", "payment"],
    "twice": ["duplicate", "multiple", "charged"],
    "tracking": ["shipment", "delivery", "courier"],
    "courier": ["delivery", "shipment", "tracking"],
    "otp": ["verification", "login", "account"],
    "email": ["account", "login", "verification"],
    "cancel": ["cancellation", "refund"],
    "cancelled": ["cancellation", "refund"],
    "cancellation": ["cancel", "refund"],
}


def tokenize(text: str) -> list[str]:
    """Return searchable lowercase words, excluding common stop words."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return [word for word in words if word not in STOP_WORDS]


def expand_query(tokens: list[str]) -> set[str]:
    """Add known customer-language synonyms without replacing original terms."""
    expanded = set(tokens)
    for word in tokens:
        expanded.update(SYNONYMS.get(word, []))
    return expanded


def load_kb_passages() -> list[dict[str, str]]:
    """Load Markdown files into section-aware passages.

    Each passage retains its filename and heading path so the agent can cite
    the precise policy section instead of only naming a whole document.
    """
    passages: list[dict[str, str]] = []

    if not KNOWLEDGE_BASE_DIR.exists():
        return passages

    for file_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue

        heading_stack: list[tuple[int, str]] = []
        current_blocks: list[str] = []

        def save_section() -> None:
            if not current_blocks:
                return

            heading = " > ".join(item[1] for item in heading_stack)
            body = "\n\n".join(current_blocks).strip()

            if body:
                # Keep the heading with the passage so heading terms can
                # participate in retrieval and be shown as a source citation.
                passages.append({
                    "source": file_path.name,
                    "section": heading or "General information",
                    "content": body,
                    "search_text": f"{heading}\n{body}",
                })

        for line in content.splitlines():
            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)

            if heading_match:
                save_section()
                current_blocks = []

                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                heading_stack = [
                    (old_level, old_heading)
                    for old_level, old_heading in heading_stack
                    if old_level < level
                ]
                heading_stack.append((level, heading_text))
                continue

            if not line.strip():
                continue

            current_blocks.append(line.strip())

        save_section()

    return passages


def search_kb(query: str) -> str:
    """Retrieve relevant policy sections and return source-grounded JSON."""
    if not isinstance(query, str) or not query.strip():
        return json.dumps({
            "found": False,
            "message": "Empty search query.",
            "results": [],
        })

    passages = load_kb_passages()
    if not passages:
        return json.dumps({
            "found": False,
            "message": "No company knowledge documents found.",
            "results": [],
        })

    original_tokens = tokenize(query)
    if not original_tokens:
        return json.dumps({
            "found": False,
            "message": "No searchable keywords found.",
            "results": [],
        })

    original_set = set(original_tokens)
    expanded_set = expand_query(original_tokens)

    # Count in how many passages each term occurs. Rare terms receive more
    # weight than common terms, similar to inverse document frequency.
    document_frequency = Counter()
    passage_token_data = []

    for passage in passages:
        body_tokens = tokenize(passage["content"])
        heading_tokens = tokenize(passage["section"])
        all_tokens = set(body_tokens + heading_tokens)
        passage_token_data.append((body_tokens, heading_tokens, all_tokens))

        for term in all_tokens:
            document_frequency[term] += 1

    total_passages = max(len(passages), 1)
    scored: list[dict] = []

    for passage, (body_tokens, heading_tokens, all_tokens) in zip(
        passages, passage_token_data
    ):
        original_matches = original_set.intersection(all_tokens)
        synonym_matches = (expanded_set - original_set).intersection(all_tokens)

        if not original_matches:
            continue

        # IDF rewards distinctive query terms. Heading matches get an
        # additional boost because headings describe the policy topic.
        score = 0.0
        for term in original_matches:
            idf = 1.0 + (
                total_passages / (1 + document_frequency[term])
            ) ** 0.5
            score += idf * (2.0 if term in heading_tokens else 1.0)

        for term in synonym_matches:
            idf = 1.0 + (
                total_passages / (1 + document_frequency[term])
            ) ** 0.5
            score += idf * 0.25

        coverage = len(original_matches) / max(len(original_set), 1)
        score += coverage * 2.0

        # Exact adjacent phrase matches are useful evidence, but not the
        # sole basis for accepting a passage.
        normalized_query = " ".join(original_tokens)
        normalized_content = " ".join(tokenize(passage["search_text"]))
        if len(original_tokens) > 1 and normalized_query in normalized_content:
            score += 2.0

        scored.append({
            "source": passage["source"],
            "section": passage["section"],
            "passage": passage["content"],
            "relevance_score": round(score, 3),
            "query_coverage": round(coverage, 3),
            "matched_terms": sorted(original_matches),
        })

    if not scored:
        return json.dumps({
            "found": False,
            "message": (
                "No relevant policy section matched the customer's "
                "original keywords. Do not infer an answer."
            ),
            "results": [],
        })

    scored.sort(
        key=lambda item: (
            item["relevance_score"],
            item["query_coverage"],
        ),
        reverse=True,
    )

    # Reject weak one-word matches for longer questions. Short, focused
    # questions can still retrieve a section with one meaningful match.
    meaningful_matches_required = 1 if len(original_set) <= 2 else 2

    relevant = [
        item for item in scored
        if len(item["matched_terms"]) >= meaningful_matches_required
        and item["query_coverage"] >= (
            0.25 if len(original_set) <= 2 else 0.20
        )
    ]

    if not relevant:
        return json.dumps({
            "found": False,
            "message": (
                "No sufficiently relevant policy evidence was found. "
                "Do not answer from assumptions; refer the case to support."
            ),
            "results": [],
        })

    # Avoid returning several nearly identical sections from one file when
    # other policy documents contain distinct, useful evidence.
    selected: list[dict] = []
    seen_sections: set[tuple[str, str]] = set()
    limit = max(1, min(int(MAX_KB_RESULTS), 3))

    for item in relevant:
        identity = (item["source"], item["section"])
        if identity in seen_sections:
            continue
        selected.append(item)
        seen_sections.add(identity)
        if len(selected) >= limit:
            break

    return json.dumps({
        "found": True,
        "results": selected,
    }, indent=2)


KNOWLEDGE_AGENT_PROMPT = """
You are Kovai Cart's Knowledge Base Agent.

RESPONSIBILITY:
Retrieve policy evidence from the local Markdown knowledge base by using
the search_kb Python tool. You do not write the customer-facing response.

RETRIEVAL RULES:
1. Always call search_kb.
2. Search the customer's original complaint. Include the QueryAgent summary
   in the search query when it adds useful context.
3. Use only the passages returned by search_kb as company-policy evidence.
4. Return no more than 3 passages.
5. Preserve each passage's exact source filename and section heading.
6. Do not treat a relevance score as proof that a policy applies.
7. If found is false, report NOT_FOUND and do not infer a policy.
8. If evidence covers only part of the question, clearly identify what is
   supported and what is not covered.
9. Never invent eligibility, exceptions, refund approvals, timelines,
   inventory, tracking updates, or actions performed by the company.
10. Do not use general knowledge as a substitute for company policy.
11. Do not make urgency, sentiment, or escalation decisions.
12. Pass the evidence, filenames, and section headings clearly to
    ResolutionAgent.

OUTPUT FORMAT:

KNOWLEDGE_STATUS:
FOUND or NOT_FOUND

RETRIEVED_PASSAGES:
For each result, include:
- Source filename
- Section heading
- Retrieved policy passage

SOURCES:
List unique source filenames and their section headings, or None.

If the knowledge base does not contain enough evidence to answer the
customer's question, explicitly report NOT_FOUND or PARTIALLY_FOUND.
"""


def create_knowledge_agent(
    model_client: OpenAIChatCompletionClient,
) -> AssistantAgent:
    """Create the knowledge agent with its local retrieval tool."""
    return AssistantAgent(
        name="KnowledgeAgent",
        description=(
            "Retrieves relevant company policy sections with source "
            "filenames and heading-level citations."
        ),
        model_client=model_client,
        tools=[search_kb],
        system_message=KNOWLEDGE_AGENT_PROMPT,
        reflect_on_tool_use=True,
    )
