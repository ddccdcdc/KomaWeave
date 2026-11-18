"""Context-aware, token-efficient multi-provider AI translator for manga & manhwa."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import LANGUAGES
from app.provider_store import PROVIDERS, get_provider_secret, validate_base_url

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)
_INDEXED_LINE_RE = re.compile(r"^\s*\[?(\d+)\]?\s*(?:\||｜|:|\)|-)\s*(.*)$")

_PURE_PUNCT_RE = re.compile(r"^[\s\d\.\,\!\?\-\—\–\…\~\*\(\)\[\]\{\}\:\;\'\"\/\\]+$")


class AITranslator:
    """Supports OpenAI, Gemini, DeepSeek, Anthropic, Groq, OpenRouter, and Custom OpenAI-compatible APIs."""

    def __init__(self, timeout: Optional[float] = None, max_concurrency: int = 16) -> None:
        self._timeout = float(timeout if timeout is not None else os.getenv("AI_TRANSLATE_TIMEOUT", "120.0"))
        self._sem = asyncio.Semaphore(max_concurrency)
        self._cache: Dict[Tuple[str, str, str, str, str], str] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
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

    @classmethod
    def _extract_indexed_translations(cls, raw_text: str, expected_len: int) -> Dict[int, str]:
        """Parse compact `ID|translation` lines, falling back to JSON if needed."""
        txt = raw_text.strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```(?:json|text|tsv)?\s*|\s*```$", "", txt, flags=re.DOTALL).strip()

        found: Dict[int, str] = {}
        last_idx: Optional[int] = None
        for line in txt.splitlines():
            m = _INDEXED_LINE_RE.match(line)
            if m:
                idx = int(m.group(1))
                val = m.group(2).strip()
                if 0 <= idx < expected_len:
                    found[idx] = val
                    last_idx = idx
                    continue
            stripped = line.strip()
            if stripped and last_idx is not None and stripped not in {"{", "}", "[", "]"}:
                found[last_idx] = f"{found[last_idx]} {stripped}".strip()

        if found:
            return found

        arr = cls._extract_translations(txt, expected_len)
        if arr is not None:
            return {i: s for i, s in enumerate(arr)}
        return {}

    async def _call_provider(
        self,
        provider_id: str,
        model: str,
        api_key: str,
        base_url: str,
        lines: List[str],
        target_lang: str,
        source_lang: str,
    ) -> Dict[int, str]:
        lang_name = LANGUAGES.get(target_lang, target_lang)
        src_hint = f" from {LANGUAGES.get(source_lang, source_lang)}" if source_lang != "auto" else ""
        sys_prompt = (
            f"Manga translator{src_hint}->{lang_name}. "
            "Keep natural tone, gender & SFX; fix minor OCR typos. "
            f"Output ONLY lines 'ID|translation' for IDs 0..{len(lines) - 1}. No quotes, JSON, or extra text."
        )
        user_payload = "\n".join(
            f"{i}|{txt.replace(chr(10), ' ').strip()}" for i, txt in enumerate(lines)
        )
        max_out = min(4096, max(256, int(len(user_payload) * 2.5) + 128))
        if PROVIDERS.get(provider_id, {}).get("supports_custom_url"):
            base_url = validate_base_url(base_url)
        client = await self._get_client()

        async with self._sem:
            if provider_id == "gemini":
                gen_cfg: Dict[str, Any] = {
                    "temperature": 0.1,
                    "maxOutputTokens": max_out,
                }
                mdl_lower = model.lower()
                if "2.5" in mdl_lower and "pro" not in mdl_lower:
                    gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
                payload_gem = {
                    "systemInstruction": {"parts": [{"text": sys_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_payload}]}],
                    "generationConfig": gen_cfg,
                }
                resp = await client.post(
                    f"{base_url}/models/{model}:generateContent",
                    params={"key": api_key},
                    json=payload_gem,
                )
                if resp.status_code == 400 and "thinkingConfig" in gen_cfg:
                    gen_cfg.pop("thinkingConfig", None)
                    resp = await client.post(
                        f"{base_url}/models/{model}:generateContent",
                        params={"key": api_key},
                        json=payload_gem,
                    )
                resp.raise_for_status()
                data = resp.json()
                raw = data["candidates"][0]["content"]["parts"][0]["text"]
                return self._extract_indexed_translations(raw, len(lines))

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
                        "max_tokens": max_out,
                        "temperature": 0.1,
                        "system": sys_prompt,
                        "messages": [{"role": "user", "content": user_payload}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["content"][0]["text"]
                return self._extract_indexed_translations(raw, len(lines))

            # OpenAI-compatible: openai, deepseek, groq, openrouter, custom
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            body: Dict[str, Any] = {
                "model": model,
                "temperature": 0.1,
                "max_tokens": max_out,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_payload},
                ],
            }

            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            return self._extract_indexed_translations(raw, len(lines))

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
        unique_map: Dict[str, List[int]] = {}
        for i, t in enumerate(texts):
            clean = t.strip()
            ckey = (provider_id, mdl, source_lang, target_lang, clean)
            if not clean:
                results[i] = ""
            elif _PURE_PUNCT_RE.match(clean):
                results[i] = clean
            elif ckey in self._cache:
                results[i] = self._cache[ckey]
            else:
                unique_map.setdefault(clean, []).append(i)

        if not unique_map:
            return [r or "" for r in results]

        # Batch ~28 blocks (~1800 chars) per request so multi-page chapters parallelize across concurrent streams
        chunks: List[Tuple[List[List[int]], List[str]]] = []
        cur_idx: List[List[int]] = []
        cur_txt: List[str] = []
        cur_len = 0
        for txt, idxs in unique_map.items():
            if cur_txt and (len(cur_txt) >= 28 or cur_len + len(txt) > 1800):
                chunks.append((cur_idx, cur_txt))
                cur_idx, cur_txt, cur_len = [], [], 0
            cur_idx.append(idxs)
            cur_txt.append(txt)
            cur_len += len(txt) + 5
        if cur_txt:
            chunks.append((cur_idx, cur_txt))

        async def _run_chunk(c_idx: List[List[int]], c_txt: List[str]) -> List[Tuple[List[int], str, str]]:
            last_exc: Optional[Exception] = None
            parsed_map: Dict[int, str] = {}
            for attempt in range(2):
                try:
                    parsed_map = await self._call_provider(
                        provider_id, mdl, key, url, c_txt, target_lang, source_lang
                    )
                    last_exc = None
                    break
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_exc = exc
                    if attempt == 0:
                        await asyncio.sleep(1.0)
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    if attempt == 0 and exc.response.status_code in {429, 500, 502, 503, 504}:
                        await asyncio.sleep(1.5)
                    else:
                        raise
            if last_exc is not None:
                raise last_exc
            if len(parsed_map) == len(c_txt):
                return [(c_idx[i], c_txt[i], parsed_map[i]) for i in range(len(c_txt))]
            if 0 < len(parsed_map) < len(c_txt):
                # Keep all translated items and retry only missing indices
                resolved = [(c_idx[i], c_txt[i], parsed_map[i]) for i in parsed_map]
                missing = [i for i in range(len(c_txt)) if i not in parsed_map]
                if len(missing) < len(c_txt):
                    rem = await _run_chunk([c_idx[i] for i in missing], [c_txt[i] for i in missing])
                    return resolved + rem
            if len(c_txt) == 1:
                return [(c_idx[0], c_txt[0], parsed_map.get(0, c_txt[0]))]
            mid = len(c_txt) // 2
            left, right = await asyncio.gather(
                _run_chunk(c_idx[:mid], c_txt[:mid]),
                _run_chunk(c_idx[mid:], c_txt[mid:]),
            )
            return left + right

        chunk_outputs = await asyncio.gather(*(_run_chunk(ci, ct) for ci, ct in chunks))
        for out in chunk_outputs:
            for idxs, src, dst in out:
                self._cache[(provider_id, mdl, source_lang, target_lang, src)] = dst
                for idx in idxs:
                    results[idx] = dst

        return [r or "" for r in results]

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
