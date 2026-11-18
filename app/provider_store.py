"""Secure local provider & API key store + live model discovery."""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from app.config import ALLOW_LOCAL_PROVIDER_URLS, BASE_DIR

SECRETS_PATH = BASE_DIR / ".secrets.json"

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "google_free": {
        "id": "google_free",
        "name": "Google Translate (Free)",
        "requires_key": False,
        "supports_custom_url": False,
        "base_url": "",
        "default_model": "google-translate",
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI (ChatGPT)",
        "requires_key": True,
        "supports_custom_url": False,
        "base_url": "https://api.openai.com/v1",
        "default_model": "",
    },
    "gemini": {
        "id": "gemini",
        "name": "Google Gemini",
        "requires_key": True,
        "supports_custom_url": False,
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "",
    },
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek",
        "requires_key": True,
        "supports_custom_url": False,
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "",
    },
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic (Claude)",
        "requires_key": True,
        "supports_custom_url": False,
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "",
    },
    "groq": {
        "id": "groq",
        "name": "Groq",
        "requires_key": True,
        "supports_custom_url": False,
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "",
    },
    "openrouter": {
        "id": "openrouter",
        "name": "OpenRouter",
        "requires_key": True,
        "supports_custom_url": False,
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "",
    },
    "custom": {
        "id": "custom",
        "name": "Custom (OpenAI-Compatible)",
        "requires_key": True,
        "supports_custom_url": True,
        "base_url": "",
        "default_model": "",
    },
}


def _load_raw() -> Dict[str, Dict[str, str]]:
    if not SECRETS_PATH.exists():
        return {}
    try:
        data = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_raw(data: Dict[str, Dict[str, str]]) -> None:
    SECRETS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(SECRETS_PATH, 0o600)
    except OSError:
        pass


def _is_disallowed_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address, allow_loopback: bool) -> bool:
    if ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_unspecified or ip_obj.is_reserved:
        return True
    if ip_obj.is_loopback:
        return not allow_loopback
    if ip_obj.is_private:
        return True
    return False


def validate_base_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValueError("base_url must be a valid http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not contain embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain query parameters or fragments.")

    host = parsed.hostname.lower()
    allow_local = ALLOW_LOCAL_PROVIDER_URLS or (os.getenv("KOMAWEAVE_ALLOW_LOCAL_PROVIDERS", "0") == "1")
    is_explicit_localhost = host in {"localhost", "127.0.0.1", "::1"}
    if not (allow_local and is_explicit_localhost):
        if host in {"localhost", "127.0.0.1", "::1", "metadata.google.internal"}:
            raise ValueError(
                "Loopback/local provider URLs are blocked by SSRF policy. "
                "Set KOMAWEAVE_ALLOW_LOCAL_PROVIDERS=1 to allow localhost."
            )

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve provider host '{host}': {exc}") from exc
    if not infos:
        raise ValueError(f"No IP addresses resolved for provider host '{host}'.")

    for info in infos:
        raw_ip = info[4][0].split("%")[0]
        try:
            ip_obj = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise ValueError(f"Invalid IP address resolved for '{host}'.") from exc
        if _is_disallowed_ip(ip_obj, allow_loopback=(allow_local and is_explicit_localhost)):
            raise ValueError(
                f"Blocked provider IP '{ip_obj}' (loopback, private, link-local, or metadata addresses are not allowed)."
            )

    return cleaned


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def get_provider_secret(provider_id: str) -> Dict[str, str]:
    raw = _load_raw()
    return raw.get(provider_id, {})


def save_provider_secret(
    provider_id: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    selected_model: Optional[str] = None,
) -> Dict[str, Any]:
    if provider_id not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_id}")
    raw = _load_raw()
    entry = raw.get(provider_id, {})
    if api_key is not None and api_key.strip():
        entry["api_key"] = api_key.strip()
    if base_url is not None:
        if not PROVIDERS[provider_id].get("supports_custom_url") and base_url.strip():
            raise ValueError(f"Provider '{provider_id}' does not support a custom base_url.")
        entry["base_url"] = validate_base_url(base_url)
    if selected_model is not None:
        entry["selected_model"] = selected_model.strip()
    raw[provider_id] = entry
    _save_raw(raw)
    return get_public_provider_info(provider_id)


def delete_provider_key(provider_id: str) -> Dict[str, Any]:
    raw = _load_raw()
    if provider_id in raw and "api_key" in raw[provider_id]:
        del raw[provider_id]["api_key"]
        _save_raw(raw)
    return get_public_provider_info(provider_id)


def get_public_provider_info(provider_id: str) -> Dict[str, Any]:
    meta = PROVIDERS[provider_id]
    saved = get_provider_secret(provider_id)
    key = saved.get("api_key", "")
    return {
        **meta,
        "base_url": saved.get("base_url") or meta["base_url"],
        "selected_model": saved.get("selected_model") or meta["default_model"],
        "has_key": bool(key),
        "masked_key": _mask_key(key),
    }


def get_all_public_providers() -> List[Dict[str, Any]]:
    return [get_public_provider_info(pid) for pid in PROVIDERS]


async def fetch_provider_models(
    provider_id: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[str]:
    """Query the provider's live models endpoint and return sorted chat/text model IDs."""
    if provider_id not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_id}")
    if provider_id == "google_free":
        return ["google-translate"]

    saved = get_provider_secret(provider_id)
    key = (api_key or "").strip() or saved.get("api_key", "")
    if PROVIDERS[provider_id].get("supports_custom_url"):
        raw_url = (base_url or "").strip().rstrip("/") or saved.get("base_url") or PROVIDERS[provider_id]["base_url"]
        url = validate_base_url(raw_url)
    else:
        url = PROVIDERS[provider_id]["base_url"]

    if not url:
        raise ValueError("base_url is required for custom provider.")
    if not key and provider_id != "custom":
        raise ValueError("API key is required to fetch models.")

    async with httpx.AsyncClient(timeout=12.0, follow_redirects=False) as client:
        if provider_id == "gemini":
            resp = await client.get(f"{url}/models", params={"key": key})
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    name = m.get("name", "")
                    if name.startswith("models/"):
                        name = name[len("models/"):]
                    models.append(name)
            return sorted(set(models), reverse=True)

        if provider_id == "anthropic":
            resp = await client.get(
                f"{url}/models",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return sorted({m["id"] for m in data.get("data", []) if "id" in m}, reverse=True)

        # OpenAI-compatible (/models)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = await client.get(f"{url}/models", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        raw_ids = [m["id"] for m in data.get("data", []) if "id" in m]

        if provider_id == "openai":
            skip = ("embedding", "whisper", "tts", "dall-e", "moderation", "babbage", "davinci", "realtime", "audio")
            raw_ids = [mid for mid in raw_ids if not any(s in mid.lower() for s in skip)]

        return sorted(set(raw_ids))
