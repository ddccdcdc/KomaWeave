"""Context-aware, token-efficient multi-provider AI translator for manga & manhwa."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import LANGUAGES
from app.provider_store import PROVIDERS, get_provider_secret

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)


class AITranslator:
    """Supports OpenAI, Gemini, DeepSeek, Anthropic, Groq, OpenRouter, and Custom OpenAI-compatible APIs."""

    def __init__(self, timeout: float = 30.0, max_concurrency: int = 6) -> None:
        self._timeout = timeout
        self._sem = asyncio.Semaphore(max_concurrency)
        self._cache: Dict[Tuple[str, str, str, str, str], str] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=24, max_keepalive_connections=12),
            )
        return self._client

    @staticmethod
    def _extract_translations(raw_text: str, expected_len: int) -> Optional[List[str]]:
        txt = raw_text.strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.DOTALL).strip()

        parsed: Any = None
        try:
            parsed = json.loads(txt)
        except Exception:
            m_obj = _JSON_OBJ_RE.search(txt)
            m_arr = _JSON_ARR_RE.search(txt)
            for m in (m_obj, m_arr):
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        break
                    except Exception:
                        pass

        if isinstance(parsed, dict):
            arr = parsed.get("t") or parsed.get("translations") or next(
                (v for v in parsed.values() if isinstance(v, list)), None
            )
            if isinstance(arr, list) and len(arr) == expected_len:
                return [str(x).strip() for x in arr]
        elif isinstance(parsed, list) and len(parsed) == expected_len:
            return [str(x).strip() for x in parsed]
        return None

    async def _call_provider(
        self,
        provider_id: str,
        model: str,
        api_key: str,
        base_url: str,
        lines: List[str],
        target_lang: str,
        source_lang: str,
    ) -> Optional[List[str]]:
        lang_name = LANGUAGES.get(target_lang, target_lang)
        src_hint = f" from {LANGUAGES.get(source_lang, source_lang)}" if source_lang != "auto" else ""
        sys_prompt = (
            f"Manga/manhwa translator{src_hint} to {lang_name}. "
            "Use full scene context: preserve character gender, pronouns & tone; "
            "adapt SFX naturally; fix obvious OCR typos. "
            'Reply ONLY JSON: {"t":[...]} matching exact input count & order.'
        )
        user_payload = json.dumps(lines, ensure_ascii=False)
        client = await self._get_client()

        async with self._sem:
            if provider_id == "gemini":
                resp = await client.post(
                    f"{base_url}/models/{model}:generateContent",
                    params={"key": api_key},
                    json={
                        "systemInstruction": {"parts": [{"text": sys_prompt}]},
                        "contents": [{"role": "user", "parts": [{"text": user_payload}]}],
                        "generationConfig": {
                            "temperature": 0.2,
                            "responseMimeType": "application/json",
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["candidates"][0]["content"]["parts"][0]["text"]
                return self._extract_translations(raw, len(lines))

            if provider_id == "anthropic":
                resp = await client.post(
                    f"{base_url}/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": min(4096, max(512, len(user_payload) * 3)),
                        "temperature": 0.2,
                        "system": sys_prompt,
                        "messages": [{"role": "user", "content": user_payload}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["content"][0]["text"]
                return self._extract_translations(raw, len(lines))

            # OpenAI-compatible: openai, deepseek, groq, openrouter, custom
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            body: Dict[str, Any] = {
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_payload},
                ],
            }
            if provider_id in {"openai", "deepseek", "groq"}:
                body["response_format"] = {"type": "json_object"}

            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            return self._extract_translations(raw, len(lines))

    async def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: str = "auto",
        provider_id: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> List[str]:
        if not texts:
            return []
        if provider_id not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider_id}")

        saved = get_provider_secret(provider_id)
        key = (api_key or "").strip() or saved.get("api_key", "")
        url = (base_url or "").strip().rstrip("/") or saved.get("base_url") or PROVIDERS[provider_id]["base_url"]
        mdl = (model or "").strip() or saved.get("selected_model") or PROVIDERS[provider_id]["default_model"]

        if not key and PROVIDERS[provider_id]["requires_key"] and provider_id != "custom":
            raise ValueError(f"API key is not configured for {PROVIDERS[provider_id]['name']}.")
        if not mdl:
            raise ValueError(f"No model selected for {PROVIDERS[provider_id]['name']}.")

        results: List[Optional[str]] = [None] * len(texts)
        uncached_idx: List[int] = []
        uncached_txt: List[str] = []
        for i, t in enumerate(texts):
            clean = t.strip()
            ckey = (provider_id, mdl, source_lang, target_lang, clean)
            if not clean:
                results[i] = ""
            elif ckey in self._cache:
                results[i] = self._cache[ckey]
            else:
                uncached_idx.append(i)
                uncached_txt.append(clean)

        if not uncached_txt:
            return [r or "" for r in results]

        # Aggregate into large context-aware chunks (<=60 blocks & <=4500 chars)
        chunks: List[Tuple[List[int], List[str]]] = []
        cur_idx: List[int] = []
        cur_txt: List[str] = []
        cur_len = 0
        for idx, txt in zip(uncached_idx, uncached_txt):
            if cur_txt and (len(cur_txt) >= 60 or cur_len + len(txt) > 4500):
                chunks.append((cur_idx, cur_txt))
                cur_idx, cur_txt, cur_len = [], [], 0
            cur_idx.append(idx)
            cur_txt.append(txt)
            cur_len += len(txt) + 4
        if cur_txt:
            chunks.append((cur_idx, cur_txt))

        async def _run_chunk(c_idx: List[int], c_txt: List[str]) -> List[Tuple[int, str, str]]:
            parsed = await self._call_provider(
                provider_id, mdl, key, url, c_txt, target_lang, source_lang
            )
            if parsed is not None and len(parsed) == len(c_txt):
                return list(zip(c_idx, c_txt, parsed))
            if len(c_txt) == 1:
                return [(c_idx[0], c_txt[0], c_txt[0])]
            mid = len(c_txt) // 2
            left, right = await asyncio.gather(
                _run_chunk(c_idx[:mid], c_txt[:mid]),
                _run_chunk(c_idx[mid:], c_txt[mid:]),
            )
            return left + right

        chunk_outputs = await asyncio.gather(*(_run_chunk(ci, ct) for ci, ct in chunks))
        for out in chunk_outputs:
            for idx, src, dst in out:
                self._cache[(provider_id, mdl, source_lang, target_lang, src)] = dst
                results[idx] = dst

        return [r or "" for r in results]

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
