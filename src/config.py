"""Configuration for the benchmark."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
QUESTIONS_DIR = PROJECT_ROOT / "yks_2025"
RESULTS_DIR = PROJECT_ROOT / "results"
ANSWERS_FILE = PROJECT_ROOT / "answers.json"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model identifiers
OPENAI_MODEL = "gpt-5-2025-08-07"
CLAUDE_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.5-flash"

# GitHub image URLs (always use URLs for cost efficiency)
GITHUB_IMAGE_BASE_URL = "https://raw.githubusercontent.com/adarkeser/yks-ai-benchmark/main/yks_2025"

# Batch API settings
POLL_INTERVAL = 60  # seconds
MAX_WAIT_TIME = 86400  # 24 hours in seconds

# Pricing (per million tokens) - 50% batch discount already applied
PRICING = {
    "chatgpt-5": {
        "input": 0.625,  # TBD - to be updated with actual pricing
        "output": 5,
    },
    "claude-sonnet-4-5": {
        "input": 2.50,
        "output": 12.50,
    },
    "gemini-3-pro": {
    "input": 1.00,   # You'll use the ≤200k tier
    "output": 6.00,  # You'll use the ≤200k tier
    }
}

