
"""Central configuration for Kovai Cart AI Support."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project directories
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
LOGS_DIR = BASE_DIR / "logs"
TICKETS_FILE = LOGS_DIR / "tickets.csv"

# Load environment variables
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. Add it to your project's .env file."
    )

# Lightweight model for fast and economical agent execution
MODEL_NAME = "gpt-4o-mini"

# AutoGen safety settings
MAX_MESSAGES = 12
MAX_KB_RESULTS = 3

# Ensure runtime directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
