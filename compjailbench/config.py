"""
Configuration for CompJailBench.

Values are loaded from environment variables (via a local .env file, using
python-dotenv) rather than hardcoded, since this file gets committed/shared
and should never contain real credentials.

LLM_PROVIDER controls which backend get_client() returns:
    "mock"   - no credentials needed, canned responses (see client.py)
    "gemini" - Google AI Studio, free, no credit card required
    "azure"  - Azure OpenAI (needs a card-verified Azure account)

Create a `.env` file in the project root. For Gemini (recommended while
you don't have Azure access yet):

    LLM_PROVIDER=gemini
    GEMINI_API_KEY=<get for free at https://aistudio.google.com/app/apikey>
    GEMINI_MODEL=gemini-2.0-flash

For Azure, once you have it:

    LLM_PROVIDER=azure
    AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/
    AZURE_API_KEY=<your-key>
    AZURE_API_VERSION=2024-02-15-preview
    AZURE_DEPLOYMENT=<your-deployment-name>
"""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")

# --- Azure OpenAI ---
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "")
AZURE_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT", "")

_REQUIRED_AZURE = {
    "AZURE_ENDPOINT": AZURE_ENDPOINT,
    "AZURE_API_KEY": AZURE_API_KEY,
    "AZURE_API_VERSION": AZURE_API_VERSION,
    "AZURE_DEPLOYMENT": AZURE_DEPLOYMENT,
}


def validate_azure_config():
    missing = [name for name, value in _REQUIRED_AZURE.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing required Azure config values: "
            + ", ".join(missing)
            + ". Set them in a .env file or as environment variables."
        )


# --- Google Gemini (AI Studio) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def validate_gemini_config():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Get a free key (no card required) at "
            "https://aistudio.google.com/app/apikey and set it in your .env file."
        )