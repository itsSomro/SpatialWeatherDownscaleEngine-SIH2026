import os
from typing import Dict, List, Optional

try:
    from huggingface_hub import InferenceClient
    _HF_SDK_AVAILABLE = True
except ImportError:
    _HF_SDK_AVAILABLE = False


def _get_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGINGFACE_API_KEY")
    )


def _get_model() -> Optional[str]:
    model = os.environ.get("HF_MODEL")
    return model.strip() if model and model.strip() else None


def is_configured() -> bool:
    """True only if BOTH a Hugging Face token and a model are set."""
    token = _get_token()
    return bool(token and token.strip()) and bool(_get_model())


def get_client():
    """Build an InferenceClient for the configured model, or None if not ready."""
    if not _HF_SDK_AVAILABLE:
        return None
    token = _get_token()
    model = _get_model()
    if not token or not token.strip() or not model:
        return None
    try:
        return InferenceClient(model=model, token=token.strip(), provider="auto")
    except Exception:
        return None


def chat(messages: List[Dict[str, str]]) -> str:
    """
    Send a chat-completion request. Always returns a string (never raises) —
    on any failure it returns a human-readable explanation instead.
    """
    if not _HF_SDK_AVAILABLE:
        return (
            "⚠️ The `huggingface_hub` package isn't installed. Run "
            "`pip install huggingface_hub` and restart the app."
        )

    token = _get_token()
    model = _get_model()

    if not token or not token.strip():
        return (
            "⚠️ No Hugging Face token found. Add `HF_TOKEN=hf_xxxxx` to your `.env` file "
            "(get a free token at https://huggingface.co/settings/tokens) and restart the app."
        )
    if not model:
        return (
            "⚠️ No Hugging Face model configured. Add `HF_MODEL=<org/model-name>` to your "
            "`.env` file (e.g. `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct`) and restart the app. "
            "There is no default model — you choose which one to use."
        )

    client = get_client()
    if client is None:
        return "⚠️ Could not initialize the Hugging Face client. Double-check HF_TOKEN and HF_MODEL and try again."

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=700,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "The model returned an empty response — please try again."
    except Exception as e:
        return (
            f"⚠️ The AI Data Agent couldn't reach the Hugging Face inference endpoint "
            f"(`{model}`): {e}\n\n"
            "This usually means the model is still loading (first request can take ~20s), "
            "the free inference quota was hit, the token lacks Inference Provider access, "
            "or `HF_MODEL` doesn't match a model available via Inference Providers."
        )