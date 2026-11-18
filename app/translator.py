"""Fast async Google Translate client with connection pooling & caching."""

from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional, Tuple

import httpx

_PURE_PUNCT_RE = re.compile(r"^[\s\d\.\,\!\?\-\—\–\…\~\*\(\)\[\]\{\}\:\;\'\"\/\\！？…・。、]+$")


class FastGoogleTranslator:
    """Concurrent Google Translate client with dual-endpoint fallback."""

    PRIMARY_ENDPOINT = "https://clients5.google.com/translate_a/t"
    FALLBACK_ENDPOINT = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, timeout: float = 10.0, max_concurrency: int = 12) -> None:
        self._timeout = timeout
        self._sem = asyncio.Semaphore(max_concurrency)
        self._cache: Dict[Tuple[str, str, str], str] = {}
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                headers={"User-Agent": "Mozilla/5.0"},
            )
        return self._client

    @staticmethod
    def _parse_clients5(data) -> str:
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, list) and first:
                return str(first[0]).strip()
            if isinstance(first, str):
                return first.strip()
        return ""

    async def translate_one(
        self, text: str, target_lang: str, source_lang: str = "auto"
    ) -> str:
        clean = text.strip()
        if not clean:
            return ""

        key = (source_lang, target_lang, clean)
        if key in self._cache:
            return self._cache[key]

        client = await self._get_client()
        for attempt in range(2):
            async with self._sem:
                # 1. Fast primary endpoint (clients5.google.com/translate_a/t)
                try:
                    resp = await client.get(
                        self.PRIMARY_ENDPOINT,
                        params={
                            "client": "dict-chrome-ex",
                            "sl": source_lang,
                            "tl": target_lang,
                            "q": clean,
                        },
                    )
                    if resp.status_code == 200:
                        translated = self._parse_clients5(resp.json())
                        if translated:
                            self._cache[key] = translated
                            return translated
                except Exception:
                    pass

                # 2. Fallback endpoint (translate.googleapis.com/translate_a/single)
                try:
                    resp = await client.get(
                        self.FALLBACK_ENDPOINT,
                        params={
                            "client": "gtx",
                            "sl": source_lang,
                            "tl": target_lang,
                            "dt": "t",
                            "q": clean,
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        translated = "".join(
                            seg[0] for seg in (data[0] or []) if seg and seg[0]
                        ).strip()
                        if translated:
                            self._cache[key] = translated
                            return translated
                except Exception:
                    pass
            if attempt == 0:
                await asyncio.sleep(0.25)
        return clean

    async def translate_batch(
        self, texts: List[str], target_lang: str, source_lang: str = "auto"
    ) -> List[str]:
        if not texts:
            return []
        if len(texts) == 1:
            return [await self.translate_one(texts[0], target_lang, source_lang)]

        # Check cache & deduplicate uncached items + skip pure punctuation
        results: List[Optional[str]] = [None] * len(texts)
        unique_map: Dict[str, List[int]] = {}
        for i, t in enumerate(texts):
            clean = t.strip()
            key = (source_lang, target_lang, clean)
            if not clean:
                results[i] = ""
            elif _PURE_PUNCT_RE.match(clean):
                results[i] = clean
            elif key in self._cache:
                results[i] = self._cache[key]
            else:
                unique_map.setdefault(clean, []).append(i)

        if not unique_map:
            return [r or "" for r in results]

        # Chunk into <=16 unique items & <=1500 chars per request so GET URLs stay short & parallel
        chunks: List[Tuple[List[List[int]], List[str]]] = []
        cur_idx: List[List[int]] = []
        cur_txt: List[str] = []
        cur_len = 0
        for txt, idxs in unique_map.items():
            if cur_txt and (len(cur_txt) >= 16 or cur_len + len(txt) > 1500):
                chunks.append((cur_idx, cur_txt))
                cur_idx, cur_txt, cur_len = [], [], 0
            cur_idx.append(idxs)
            cur_txt.append(txt)
            cur_len += len(txt) + 8
        if cur_txt:
            chunks.append((cur_idx, cur_txt))

        sep = "\n<|||>\n"

        async def _translate_chunk(
            c_idx: List[List[int]], c_txt: List[str]
        ) -> List[Tuple[List[int], str, str]]:
            if len(c_txt) == 1:
                dst = await self.translate_one(c_txt[0], target_lang, source_lang)
                return [(c_idx[0], c_txt[0], dst)]
            joined = sep.join(c_txt)
            translated_joined = await self.translate_one(joined, target_lang, source_lang)
            parts = [p.strip() for p in translated_joined.split("<|||>")]
            if len(parts) == len(c_txt) and all(parts):
                return list(zip(c_idx, c_txt, parts))
            mid = len(c_txt) // 2
            left, right = await asyncio.gather(
                _translate_chunk(c_idx[:mid], c_txt[:mid]),
                _translate_chunk(c_idx[mid:], c_txt[mid:]),
            )
            return left + right

        chunk_outputs = await asyncio.gather(
            *(_translate_chunk(ci, ct) for ci, ct in chunks)
        )
        for out in chunk_outputs:
            for idxs, src, dst in out:
                self._cache[(source_lang, target_lang, src)] = dst
                for idx in idxs:
                    results[idx] = dst
        return [r or "" for r in results]

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
