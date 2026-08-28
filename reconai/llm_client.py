"""
Thin Gemini wrapper. The rest of the codebase only ever calls call_json(), so
swapping providers later is a change to this one file, not a rewrite.

Requires GEMINI_API_KEY in the environment (or a .env file). Reads GEMINI_MODEL,
defaulting to gemini-2.0-flash.
"""

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_model = None


def _get_model():
    global _model
    if _model is None:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
            )
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(_MODEL_NAME)
    return _model


def _extract_json(text: str) -> dict:
    """Gemini sometimes wraps JSON in ```json fences or prose; pull the object out."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    return json.loads(text)


def call_json(prompt: str, retries: int = 2) -> dict:
    """Send prompt, return parsed JSON dict. Retries on transient/parse errors."""
    model = _get_model()
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
            )
            return _extract_json(resp.text)
        except Exception as e:  # noqa: BLE001 - deliberately broad, we retry then surface
            last_err = e
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_err}")
