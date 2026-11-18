"""End-to-end Manga/Manhwa translation pipeline with parallel inpainting + translation."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.ai_translator import AITranslator
from app.image_slicer import smart_slice_vertical, stitch_vertical
from app.inpainter import SmartInpainter
from app.ocr_engine import GeneralOCREngine, TextBlock
from app.renderer import MangaRenderer, RenderOptions
from app.translator import FastGoogleTranslator


@dataclass
class PipelineResult:
    image_bgr: np.ndarray
    blocks: List[TextBlock]
    timings_ms: Dict[str, float]
    inpaint_stats: Dict[str, int]
    device: str


class MangaTranslationPipeline:
    """Orchestrates OCR -> parallel (Smart Inpainting + Translation) -> Manga Typesetting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ocr: Optional[GeneralOCREngine] = None
        self._inpainter: Optional[SmartInpainter] = None
        self._translator: Optional[FastGoogleTranslator] = None
        self._ai_translator: Optional[AITranslator] = None
        self._renderer: Optional[MangaRenderer] = None

    @property
    def ocr(self) -> GeneralOCREngine:
        if self._ocr is None:
            with self._lock:
                if self._ocr is None:
                    self._ocr = GeneralOCREngine()
        return self._ocr

    @property
    def inpainter(self) -> SmartInpainter:
        if self._inpainter is None:
            with self._lock:
                if self._inpainter is None:
                    self._inpainter = SmartInpainter(lazy=True)
        return self._inpainter

    @property
    def translator(self) -> FastGoogleTranslator:
        if self._translator is None:
            with self._lock:
                if self._translator is None:
                    self._translator = FastGoogleTranslator()
        return self._translator

    @property
    def ai_translator(self) -> AITranslator:
        if self._ai_translator is None:
            with self._lock:
                if self._ai_translator is None:
                    self._ai_translator = AITranslator()
        return self._ai_translator

    @property
    def renderer(self) -> MangaRenderer:
        if self._renderer is None:
            with self._lock:
                if self._renderer is None:
                    self._renderer = MangaRenderer()
        return self._renderer

    @property
    def active_device(self) -> str:
        if self._inpainter is not None:
            return self._inpainter.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def warmup(self, preload: Optional[str] = None) -> None:
        """Warm up selected models (PRELOAD_MODELS=all|ocr|none)."""
        mode = (preload or os.getenv("PRELOAD_MODELS", "all")).strip().lower()
        if mode == "none":
            return
        dummy = np.full((640, 480, 3), 255, dtype=np.uint8)
        cv2.putText(dummy, "TEST", (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
        self.ocr.extract(dummy)
        if mode == "all":
            self.ocr.extract_batch([dummy, dummy])
            self.inpainter.remove_text(
                dummy, [np.array([[100, 240], [320, 240], [320, 320], [100, 320]], dtype=np.float32)]
            )

    async def _translate_texts(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: str,
        provider: str = "google_free",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> List[str]:
        if not texts:
            return []
        if not provider or provider == "google_free":
            return await self.translator.translate_batch(texts, target_lang, source_lang)
        return await self.ai_translator.translate_batch(
            texts,
            target_lang=target_lang,
            source_lang=source_lang,
            provider_id=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    @staticmethod
    def _seam_has_content(top_img: np.ndarray, bot_img: np.ndarray) -> bool:
        """Return True only when non-margin content crosses the boundary between two strips."""
        if top_img.shape[1] != bot_img.shape[1]:
            return False
        h1, w = top_img.shape[:2]
        h2 = bot_img.shape[0]
        if h1 < 24 or h2 < 24 or w < 32:
            return False
        x0, x1 = int(w * 0.05), int(w * 0.95)
        b_strip = top_img[h1 - 10 : h1, x0:x1]
        t_strip = bot_img[:10, x0:x1]
        for s in (b_strip, t_strip):
            std = float(np.std(s))
            mean = float(np.mean(s))
            # Clean white or black page margin at top/bottom edge -> no bubble crosses this seam
            if std < 12.0 and (mean >= 238.0 or mean <= 18.0):
                return False
        return True

    async def process(
        self,
        image_bgr: np.ndarray,
        target_lang: str = "ar",
        source_lang: str = "auto",
        provider: str = "google_free",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mode: str = "full",
        ocr_model: str = "auto",
        inpaint_mode: str = "auto",
        render_options: Optional[RenderOptions] = None,
    ) -> PipelineResult:
        if mode not in {"full", "ocr", "clean"}:
            raise ValueError(f"Unsupported pipeline mode: {mode}. Use 'full', 'ocr', or 'clean'.")

        # Automatically slice ultra-long vertical webtoon/chapter strips at safe gutters
        max_single_h = max(2600, int(round(image_bgr.shape[1] * 1.75)))
        if image_bgr.shape[0] > max_single_h:
            results, _ = await self.process_chapter(
                [image_bgr],
                target_lang=target_lang,
                source_lang=source_lang,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                mode=mode,
                ocr_model=ocr_model,
                inpaint_mode=inpaint_mode,
                render_options=render_options,
            )
            return results[0]

        t_start = time.perf_counter()

        # 1. Text Extraction & Layout Clustering
        t0 = time.perf_counter()
        blocks, inpaint_polys = await asyncio.to_thread(
            self.ocr.extract, image_bgr, None, ocr_model
        )
        t_ocr = (time.perf_counter() - t0) * 1000.0

        if not blocks or mode == "ocr":
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return PipelineResult(
                image_bgr=image_bgr,
                blocks=blocks,
                timings_ms={"ocr": round(t_ocr, 1), "inpaint": 0.0, "translate": 0.0, "render": 0.0, "total": round(total_ms, 1)},
                inpaint_stats={"solid_filled": 0, "aot_inpainted": 0},
                device=self.active_device,
            )

        if mode == "clean":
            ts = time.perf_counter()
            cleaned_bgr, inpaint_stats = await asyncio.to_thread(
                self.inpainter.remove_text, image_bgr, inpaint_polys, inpaint_mode
            )
            t_inpaint = (time.perf_counter() - ts) * 1000.0
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return PipelineResult(
                image_bgr=cleaned_bgr,
                blocks=blocks,
                timings_ms={"ocr": round(t_ocr, 1), "inpaint": round(t_inpaint, 1), "translate": 0.0, "render": 0.0, "total": round(total_ms, 1)},
                inpaint_stats=inpaint_stats,
                device=self.active_device,
            )

        # 2. Run Smart Inpainting (GPU) and Translation (Async Network) IN PARALLEL
        t1 = time.perf_counter()
        texts = [b.text for b in blocks]

        async def _inpaint() -> tuple[np.ndarray, dict[str, int], float]:
            ts = time.perf_counter()
            cleaned, stats = await asyncio.to_thread(
                self.inpainter.remove_text, image_bgr, inpaint_polys, inpaint_mode
            )
            return cleaned, stats, (time.perf_counter() - ts) * 1000.0

        async def _translate() -> tuple[List[str], float]:
            ts = time.perf_counter()
            tr = await self._translate_texts(
                texts, target_lang, source_lang, provider, model, api_key, base_url
            )
            return tr, (time.perf_counter() - ts) * 1000.0

        (cleaned_bgr, inpaint_stats, t_inpaint), (translated_texts, t_trans) = await asyncio.gather(
            _inpaint(), _translate()
        )
        _ = time.perf_counter() - t1

        for blk, tr_txt in zip(blocks, translated_texts):
            blk.translated_text = tr_txt

        # 3. Render Translated Text
        t2 = time.perf_counter()
        final_bgr = await asyncio.to_thread(
            self.renderer.render, cleaned_bgr, blocks, target_lang, render_options
        )
        t_render = (time.perf_counter() - t2) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return PipelineResult(
            image_bgr=final_bgr,
            blocks=blocks,
            timings_ms={
                "ocr": round(t_ocr, 1),
                "inpaint": round(t_inpaint, 1),
                "translate": round(t_trans, 1),
                "render": round(t_render, 1),
                "total": round(total_ms, 1),
            },
            inpaint_stats=inpaint_stats,
            device=self.active_device,
        )

    async def process_chapter(
        self,
        images_bgr: List[np.ndarray],
        target_lang: str = "ar",
        source_lang: str = "auto",
        ocr_workers: int = 6,
        provider: str = "google_free",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        consolidated: bool = True,
        mode: str = "full",
        ocr_model: str = "auto",
        inpaint_mode: str = "auto",
        render_options: Optional[RenderOptions] = None,
    ) -> Tuple[List[PipelineResult], Dict[str, float]]:
        """High-speed parallel batch/chapter pipeline:
        0) Smart gutter-aware slicing for ultra-long vertical strips (>2600px)
        1) Batched GPU Det + parallel CPU Rec OCR across all slices
        2) Chapter-wide consolidated AI/Google translation + GPU smart inpainting in parallel
        3) Multi-threaded Pillow manga typesetting + seamless vertical stitching
        """
        if mode not in {"full", "ocr", "clean"}:
            raise ValueError(f"Unsupported pipeline mode: {mode}. Use 'full', 'ocr', or 'clean'.")
        t_start = time.perf_counter()
        if not images_bgr:
            return [], {"ocr": 0.0, "translate": 0.0, "inpaint": 0.0, "render": 0.0, "total": 0.0}

        # 0. Group consecutive same-width images when consolidated=True ONLY when content
        # actually crosses the boundary seam (webtoon strips), keeping normal pages intact.
        runs: List[Tuple[int, int]] = []
        if consolidated and len(images_bgr) > 1:
            r_start = 0
            for idx in range(1, len(images_bgr)):
                can_join = (
                    (idx - r_start) < 6
                    and self._seam_has_content(images_bgr[idx - 1], images_bgr[idx])
                )
                if not can_join:
                    runs.append((r_start, idx))
                    r_start = idx
            runs.append((r_start, len(images_bgr)))
        else:
            runs = [(i, i + 1) for i in range(len(images_bgr))]

        slice_imgs: List[np.ndarray] = []
        slice_y_offsets: List[int] = []
        run_slice_ranges: List[Tuple[int, int]] = []
        img_run_spans: List[Tuple[int, int]] = []  # (y0, y1) inside its run

        for r_start, r_end in runs:
            cur_y = 0
            for idx in range(r_start, r_end):
                h = images_bgr[idx].shape[0]
                img_run_spans.append((cur_y, cur_y + h))
                cur_y += h

            run_canvas = (
                images_bgr[r_start]
                if (r_end - r_start) == 1
                else np.vstack(images_bgr[r_start:r_end])
            )
            w_run = run_canvas.shape[1]
            max_slice_h = max(2600, int(round(w_run * 1.75)))
            target_slice_h = max(2000, int(round(w_run * 1.40)))
            min_slice_h = max(1000, int(round(w_run * 0.85)))
            s_start = len(slice_imgs)
            if run_canvas.shape[0] > max_slice_h:
                for y0, _, s_img in smart_slice_vertical(
                    run_canvas,
                    target_height=target_slice_h,
                    min_height=min_slice_h,
                    max_height=max_slice_h,
                ):
                    slice_imgs.append(s_img)
                    slice_y_offsets.append(y0)
            else:
                slice_imgs.append(run_canvas)
                slice_y_offsets.append(0)
            run_slice_ranges.append((s_start, len(slice_imgs)))
            del run_canvas

        # 1. Parallel Batched OCR across all slices
        t0 = time.perf_counter()
        ocr_outputs = await asyncio.to_thread(
            self.ocr.extract_batch, slice_imgs, ocr_workers, ocr_model
        )
        t_ocr = (time.perf_counter() - t0) * 1000.0

        # Flatten all text blocks across the chapter for consolidated context-aware translation
        flat_texts: List[str] = []
        flat_map: List[Tuple[int, int]] = []
        for slice_idx, (blocks, _) in enumerate(ocr_outputs):
            for blk_idx, blk in enumerate(blocks):
                flat_texts.append(blk.text)
                flat_map.append((slice_idx, blk_idx))

        # 2. Run Translation & Smart Inpainting (respecting mode)
        async def _translate_all() -> Tuple[List[str], float]:
            ts = time.perf_counter()
            if not flat_texts or mode != "full":
                return [], 0.0
            if consolidated:
                tr = await self._translate_texts(
                    flat_texts, target_lang, source_lang, provider, model, api_key, base_url
                )
                return tr, (time.perf_counter() - ts) * 1000.0
            # Independent per-image translation in parallel
            per_img_tasks = []
            for s_start, s_end in run_slice_ranges:
                img_texts = [b.text for si in range(s_start, s_end) for b in ocr_outputs[si][0]]
                per_img_tasks.append(
                    self._translate_texts(
                        img_texts, target_lang, source_lang, provider, model, api_key, base_url
                    )
                )
            per_img_results = await asyncio.gather(*per_img_tasks)
            flat_tr = [t for sub in per_img_results for t in sub]
            return flat_tr, (time.perf_counter() - ts) * 1000.0

        def _inpaint_sync() -> Tuple[List[Tuple[np.ndarray, Dict[str, int]]], float]:
            ts = time.perf_counter()
            out: List[Tuple[np.ndarray, Dict[str, int]]] = []
            for img, (_, polys) in zip(slice_imgs, ocr_outputs):
                if not polys or mode == "ocr":
                    out.append((img, {"solid_filled": 0, "aot_inpainted": 0}))
                else:
                    out.append(self.inpainter.remove_text(img, polys, inpaint_mode))
            return out, (time.perf_counter() - ts) * 1000.0

        (translated_flat, t_trans), (inpainted_list, t_inpaint) = await asyncio.gather(
            _translate_all(), asyncio.to_thread(_inpaint_sync)
        )

        for (slice_idx, blk_idx), tr_txt in zip(flat_map, translated_flat):
            ocr_outputs[slice_idx][0][blk_idx].translated_text = tr_txt

        # 3. Multi-threaded Rendering (skipped in ocr/clean modes)
        t2 = time.perf_counter()

        def _render_one(idx: int) -> np.ndarray:
            cleaned, _ = inpainted_list[idx]
            blocks = ocr_outputs[idx][0]
            if not blocks or mode != "full":
                return cleaned
            return self.renderer.render(cleaned, blocks, target_lang, render_options)

        def _render_all() -> List[np.ndarray]:
            with ThreadPoolExecutor(max_workers=8) as ex:
                return list(ex.map(_render_one, range(len(slice_imgs))))

        rendered_slices = await asyncio.to_thread(_render_all)
        t_render = (time.perf_counter() - t2) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        # 4. Stitch slices back per run and split back to exact original image heights
        results: List[Optional[PipelineResult]] = [None] * len(images_bgr)
        n = len(images_bgr)
        for (r_start, r_end), (s_start, s_end) in zip(runs, run_slice_ranges):
            stitched_run = stitch_vertical(rendered_slices[s_start:s_end])
            run_blocks: List[TextBlock] = []
            solid_sum = 0
            aot_sum = 0
            for si in range(s_start, s_end):
                y_off = slice_y_offsets[si]
                _, st = inpainted_list[si]
                solid_sum += st.get("solid_filled", 0)
                aot_sum += st.get("aot_inpainted", 0)
                for b in ocr_outputs[si][0]:
                    if y_off:
                        bx0, by0, bx1, by1 = b.bbox
                        b.bbox = (bx0, by0 + y_off, bx1, by1 + y_off)
                    run_blocks.append(b)

            for orig_idx in range(r_start, r_end):
                ry0, ry1 = img_run_spans[orig_idx]
                out_img = np.ascontiguousarray(stitched_run[ry0:ry1])
                img_blocks: List[TextBlock] = []
                for b in run_blocks:
                    bx0, by0, bx1, by1 = b.bbox
                    cy = 0.5 * (by0 + by1)
                    if ry0 <= cy < ry1 or (orig_idx == r_end - 1 and cy >= ry1):
                        nb = TextBlock(
                            id=len(img_blocks),
                            text=b.text,
                            translated_text=b.translated_text,
                            bbox=(bx0, max(0, by0 - ry0), bx1, max(0, by1 - ry0)),
                            polygon=[(px, py - ry0) for px, py in b.polygon],
                            confidence=b.confidence,
                            is_vertical=b.is_vertical,
                        )
                        img_blocks.append(nb)

                results[orig_idx] = PipelineResult(
                    image_bgr=out_img,
                    blocks=img_blocks,
                    timings_ms={
                        "ocr": round(t_ocr / n, 1),
                        "inpaint": round(t_inpaint / n, 1),
                        "translate": round(t_trans / n, 1),
                        "render": round(t_render / n, 1),
                        "total": round(total_ms / n, 1),
                    },
                    inpaint_stats={"solid_filled": solid_sum, "aot_inpainted": aot_sum},
                    device=self.active_device,
                )

            del stitched_run

        final_results: List[PipelineResult] = [r for r in results if r is not None]
        n_slices = len(slice_imgs)
        del slice_imgs, inpainted_list, rendered_slices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        chapter_timings = {
            "ocr": round(t_ocr, 1),
            "translate": round(t_trans, 1),
            "inpaint": round(t_inpaint, 1),
            "render": round(t_render, 1),
            "total": round(total_ms, 1),
            "per_image_avg": round(total_ms / n, 1),
            "slices_processed": n_slices,
        }
        return final_results, chapter_timings

    async def close(self) -> None:
        if self._translator is not None:
            await self._translator.close()
        if self._ai_translator is not None:
            await self._ai_translator.close()
