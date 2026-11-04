
# 🛍️ Kovai Cart – AI-Powered Customer Support Desk

## Project Overview

Kovai Cart is an AI-powered customer support system built using Python, Microsoft AutoGen and Streamlit.

It uses four specialised AI agents to classify customer complaints, retrieve relevant company policies, generate accurate customer responses and record support tickets.

The system includes a local Retrieval-Augmented Generation (RAG) knowledge base and deterministic human escalation rules to ensure reliable customer support.

## Key Features

- Four specialised AI agents working collaboratively.
- Local RAG using company Markdown documents.
- Context-aware customer response generation.
- Automatic complaint classification.
- Deterministic human escalation.
- Unique support ticket ID generation.
- CSV-based ticket management.
- Customer-friendly Streamlit interface.
- Ticket history and CSV export.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Microsoft AutoGen 0.7.5 | Multi-agent orchestration |
| OpenAI GPT-4o-mini | Language model |
| Streamlit | Customer interface |
| Markdown | Knowledge base |
| Python keyword retrieval | Local RAG |
| Pandas | Ticket history and reporting |
| CSV | Ticket storage |

## Multi-Agent Architecture

The application consists of four specialised AI agents.

### 1. Customer Query Agent

- Analyses customer complaints.
- Identifies complaint category.
- Classifies urgency.
- Detects customer sentiment.
- Produces a concise issue summary.

### 2. Knowledge Base Agent

- Searches local Markdown documents.
- Uses keyword matching and synonym expansion.
- Retrieves the three most relevant passages.
- Returns source filenames.
- Reports when relevant information is unavailable.

### 3. Resolution Agent

- Generates customer-friendly responses.
- Uses retrieved company policies.
- Avoids inventing delivery dates or refund commitments.
- Provides confidence and supporting reasoning.
- Suggests actionable next steps.

### 4. Ticket Logging Agent

- Collects the completed ticket information.
- Generates a unique ticket ID.
- Records the ticket in CSV format.
- Applies deterministic escalation rules.
- Returns the final ticket status.

## Knowledge Base and RAG

The system uses four local Markdown documents:

1. `shipping_policy.md`
2. `refund_policy.md`
3. `account_and_login.md`
4. `product_faq.md`

The Knowledge Base Agent performs lightweight keyword-based retrieval.

The retrieval process:

1. Loads the Markdown documents.
2. Splits content into passages.
3. Tokenises the customer query.
4. Expands keywords using predefined synonyms.
5. Scores passages using keyword overlap and coverage.
6. Returns up to three relevant passages.

The Resolution Agent uses these retrieved passages to generate grounded answers.

Note: This implementation uses keyword-based retrieval rather than vector embeddings.

## Human Escalation

The system uses deterministic Python rules to identify cases requiring human support.

Escalation conditions include:

- High urgency.
- Angry customer sentiment.
- Missing relevant knowledge.
- Low resolution confidence.
- Legal threats.
- Suspected fraud.
- Payment disputes.

Tickets that meet escalation conditions are marked as `Escalated`.

Other tickets are marked as `Resolved`.

## Ticket Logging

Each ticket includes:

- Unique ticket ID.
- Timestamp.
- Customer name.
- Order ID.
- Original complaint.
- Complaint category.
- Urgency.
- Sentiment.
- Knowledge source filenames.
- Customer response.
- Final status.
- Escalation reason.

Ticket records are stored in:

`logs/tickets.csv`

## Project Structure

```text
KovaiShop/
│
├── app.py
├── team.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── agents/
│   ├── __init__.py
│   ├── query_agent.py
│   ├── knowledge_agent.py
│   ├── resolution_agent.py
│   └── log_agent.py
│
├── knowledge_base/
│   ├── shipping_policy.md
│   ├── refund_policy.md
│   ├── account_and_login.md
│   └── product_faq.md
│
├── logs/
│   └── tickets.csv
│
└── screenshots/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd KovaiShop
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

Never upload your real API key to GitHub.

### 5. Start the application

```bash
streamlit run app.py
```

Open:

http://localhost:8501

## Sample Test Scenarios

| Scenario | Expected behaviour |
|---|---|
| Delayed delivery | Retrieve shipping policy and generate response |
| Refund request | Retrieve refund policy |
| Password reset issue | Retrieve account policy |
| Damaged product | Retrieve product FAQ or refund policy |
| Suspected payment fraud | Escalate to human support |
| Legal threat | Escalate to human support |
| Unsupported question | Avoid inventing policy information |

## Security and Reliability

- API credentials are loaded from environment variables.
- Customer-facing UI does not expose internal agent conversations.
- Retrieved policy passages ground customer responses.
- Escalation decisions use deterministic Python rules.
- Ticket records are maintained in CSV format.
- The system avoids inventing live tracking updates or unsupported commitments.

## Future Enhancements

- Vector-based semantic retrieval.
- Database-backed ticket management.
- Integration with live order tracking.
- Human support dashboard.
- Customer notification system.
- Authentication and role-based access.
- Automated evaluation of response quality.

## Conclusion

Kovai Cart demonstrates how a specialised multi-agent AI architecture, local RAG and deterministic business rules can work together to improve customer support.

The system aims to provide faster customer responses while maintaining policy grounding, ticket traceability and human oversight.
