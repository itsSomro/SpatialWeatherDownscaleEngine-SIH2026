import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

try:
    from huggingface_hub import InferenceClient
    _HF_SDK_AVAILABLE = True
except ImportError:
    _HF_SDK_AVAILABLE = False

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _get_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGINGFACE_API_KEY")
    )


def _get_model() -> str:
    model = os.environ.get("HF_MODEL")
    if not model or not model.strip():
        return DEFAULT_HF_MODEL
    m = model.strip()
    if m.lower() in ["qwen", "qwen3", "qwen2.5", "qwen-7b"]:
        return "Qwen/Qwen2.5-7B-Instruct"
    if m.lower() in ["qwen-72b", "qwen2.5-72b", "qwen3-72b"]:
        return "Qwen/Qwen2.5-72B-Instruct"
    return m


def _get_gemini_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY")


def is_configured() -> bool:
    """True if either a Hugging Face token or Gemini API key is available."""
    token = _get_token()
    gemini_key = _get_gemini_key()
    return bool((token and token.strip()) or (gemini_key and gemini_key.strip()))


def get_client():
    """Build an InferenceClient for the configured model, or None if not ready."""
    if not _HF_SDK_AVAILABLE:
        return None
    token = _get_token()
    model = _get_model()
    if not token or not token.strip():
        return None
    try:
        return InferenceClient(model=model, token=token.strip(), provider="auto")
    except Exception:
        return None


def _call_gemini(messages: List[Dict[str, str]]) -> Optional[str]:
    """Fallback to Google Gemini if GEMINI_API_KEY is present."""
    gemini_key = _get_gemini_key()
    if not gemini_key or not gemini_key.strip():
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=gemini_key.strip())
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"[{role}]:\n{content}")

        full_prompt = "\n\n".join(prompt_parts)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1000,
            )
        )
        return response.text.strip() if response.text else None
    except Exception:
        return None


def chat(messages: List[Dict[str, str]], telemetry: Optional[Dict[str, Any]] = None) -> str:
    """
    Send a chat request across available backends:
    1. Hugging Face Inference API (if HF_TOKEN is configured)
    2. Google Gemini API (if GEMINI_API_KEY is configured)
    3. Built-in Offline Expert Engine with transparent error notice
    """
    token = _get_token()
    model = _get_model()
    api_error_notice = None

    # 1. Try Hugging Face if token is present
    if _HF_SDK_AVAILABLE and token and token.strip():
        client = get_client()
        if client:
            try:
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=850,
                    temperature=0.3,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
            except Exception as e:
                err_str = str(e)
                if "401" in err_str or "Invalid user token" in err_str or "Unauthorized" in err_str:
                    api_error_notice = (
                        "> ⚠️ **Hugging Face Authentication Failed (401 Unauthorized):**\n"
                        "> The `HF_TOKEN` in your `.env` file is invalid or expired. "
                        "Please get a free personal access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) "
                        "or provide a `GEMINI_API_KEY`.\n\n"
                    )
                else:
                    api_error_notice = f"> ⚠️ **Hugging Face Error:** `{err_str[:120]}` (Falling back to local expert engine)\n\n"

    # 2. Try Google Gemini if configured
    gemini_reply = _call_gemini(messages)
    if gemini_reply:
        return gemini_reply

    # 3. Built-in Offline Agro-Meteorology Expert Engine
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    try:
        try:
            from scripts.ai_advisor import _generate_offline_advisory
        except ImportError:
            from ai_advisor import _generate_offline_advisory
        safe_telemetry = telemetry or {}
        offline_reply = _generate_offline_advisory(safe_telemetry, last_user_msg)
        if api_error_notice:
            return f"{api_error_notice}{offline_reply}"
        return offline_reply
    except Exception as exc:
        return f"GramVayu AI Advisory:\n\nQuery: '{last_user_msg}'\n\nNocturnal valley cold pooling and ridge thermal gradients require active monitoring across 1km panchayat boundaries."

