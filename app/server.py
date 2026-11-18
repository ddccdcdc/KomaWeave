"""FastAPI server for the Manga & Manhwa Translation System."""

from __future__ import annotations

import asyncio
import base64
import gc
import hmac
import io
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFont
from pydantic import BaseModel

from app.config import (
    FONTS_DIR,
    LANGUAGES,
    MAX_BATCH_FILES,
    MAX_BATCH_TOTAL_BYTES,
    MAX_CONCURRENT_GPU_JOBS,
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_WIDTH,
    MAX_UPLOAD_BYTES,
    MAX_WORKING_SIDE,
    MODEL_DISCOVERY_RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_PER_MINUTE,
    REQUEST_TIMEOUT_SECONDS,
    STATIC_DIR,
    TRANSLATE_RATE_LIMIT_PER_MINUTE,
)
from app.pipeline import MangaTranslationPipeline
from app.renderer import RenderOptions
from app.provider_store import (
    delete_provider_key,
    fetch_provider_models,
    get_all_public_providers,
    save_provider_secret,
)

_pipeline: Optional[MangaTranslationPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = MangaTranslationPipeline()
    _pipeline.warmup()
    yield
    if _pipeline is not None:
        await _pipeline.close()


_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
] or ["*"]

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
_gpu_semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_GPU_JOBS))
_rate_lock = threading.Lock()
_rate_buckets: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)


def _get_configured_api_key() -> str:
    return os.getenv("KOMAWEAVE_API_KEY", "").strip()


def _client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _check_rate_limit(request: Request, bucket: str, limit_per_minute: int) -> None:
    if limit_per_minute <= 0:
        return
    now = time.monotonic()
    cutoff = now - 60.0
    key = (_client_ip(request), bucket)
    with _rate_lock:
        dq = _rate_buckets[key]
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait before retrying.",
                headers={"Retry-After": "60"},
            )
        dq.append(now)


def _require_auth(request: Request) -> None:
    expected = _get_configured_api_key()
    if not expected:
        return
    provided = request.headers.get("X-API-Key", "").strip()
    if not provided:
        auth = request.headers.get("Authorization", "").strip()
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: valid X-API-Key or Authorization: Bearer token required.",
        )


def _rate_dep(bucket: str, limit: int):
    def _dep(request: Request) -> None:
        _check_rate_limit(request, bucket, limit)

    return _dep


async def _read_upload_limited(upload: UploadFile, max_bytes: int, label: str = "File") -> bytes:
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{label} exceeds maximum allowed size ({max_bytes // (1024 * 1024)} MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_and_decode_image(raw: bytes, filename: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(raw)) as pil_img:
            w, h = pil_img.size
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=413,
            detail=f"Image '{filename}' exceeds maximum allowed pixel count ({MAX_IMAGE_PIXELS:,}).",
        )
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid or unsupported image file: {filename}")

    if w <= 0 or h <= 0 or w > MAX_IMAGE_WIDTH or h > MAX_IMAGE_HEIGHT or (w * h) > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image '{filename}' dimensions ({w}x{h}) exceed limits "
                f"(max {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}, {MAX_IMAGE_PIXELS:,} pixels)."
            ),
        )

    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail=f"Invalid or unsupported image file: {filename}")

    # Gracefully downscale overly large single pages/spreads so they never freeze CPU/GPU memory
    ih, iw = img_bgr.shape[:2]
    is_tall_strip = ih > iw * 2.2
    if not is_tall_strip and max(ih, iw) > MAX_WORKING_SIDE:
        scale = MAX_WORKING_SIDE / float(max(ih, iw))
        nw = max(64, int(round(iw * scale)))
        nh = max(64, int(round(ih * scale)))
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    elif is_tall_strip and iw > MAX_WORKING_SIDE:
        scale = MAX_WORKING_SIDE / float(iw)
        nw = MAX_WORKING_SIDE
        nh = max(64, int(round(ih * scale)))
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    return img_bgr


app = FastAPI(title="KomaWeave", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _api_auth_middleware(request: Request, call_next):
    path = request.url.path
    if request.method != "OPTIONS" and path.startswith("/api/") and path != "/api/health":
        expected = _get_configured_api_key()
        if expected:
            provided = request.headers.get("X-API-Key", "").strip()
            if not provided:
                auth = request.headers.get("Authorization", "").strip()
                if auth.lower().startswith("bearer "):
                    provided = auth[7:].strip()
            if not provided or not hmac.compare_digest(provided, expected):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized: valid X-API-Key or Authorization: Bearer token required."},
                )
    return await call_next(request)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web UI not found.")
    return FileResponse(str(index_file), headers={"Cache-Control": "no-store"})


@app.get("/api/languages")
async def get_languages():
    return {"languages": LANGUAGES}


_CACHED_DML_GPU_NAME: Optional[str] = None


def _get_dml_gpu_name() -> str:
    global _CACHED_DML_GPU_NAME
    if _CACHED_DML_GPU_NAME is not None:
        return _CACHED_DML_GPU_NAME
    name = "DirectML GPU"
    if sys.platform == "win32":
        try:
            ps = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Basic Display' } | Select-Object -First 1 -ExpandProperty Name)",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=4,
                check=False,
            )
            found = ps.stdout.strip()
            if found:
                name = f"{found} (DirectML)"
        except Exception:
            pass
    _CACHED_DML_GPU_NAME = name
    return name


@app.get("/api/health")
async def health():
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ocr_loaded = _pipeline is not None and _pipeline._ocr is not None
    ocr_dml = bool(_pipeline._ocr.use_dml) if ocr_loaded else False
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif mps_avail:
        device_name = "Apple MPS"
    elif ocr_dml or (_pipeline is not None and _pipeline._inpainter is not None and _pipeline._inpainter.device_label == "dml"):
        device_name = _get_dml_gpu_name()
    else:
        device_name = "CPU"
    aot_loaded = (
        _pipeline is not None
        and _pipeline._inpainter is not None
        and (
            _pipeline._inpainter._model is not None
            or _pipeline._inpainter._ort_session is not None
        )
    )
    return {
        "status": "ready" if _pipeline is not None else "initializing",
        "version": app.version,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps_avail,
        "ocr_loaded": ocr_loaded,
        "aot_loaded": aot_loaded,
        "ocr_cuda": bool(_pipeline._ocr.use_cuda) if ocr_loaded else False,
        "ocr_dml": ocr_dml,
        "device": device_name,
        "auth_required": bool(_get_configured_api_key()),
    }


@app.get("/api/fonts")
async def list_fonts():
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    fonts = sorted(
        p.name for p in FONTS_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".ttf", ".otf"}
    )
    return {"fonts": fonts}


@app.post("/api/fonts", dependencies=[Depends(_rate_dep("api", RATE_LIMIT_PER_MINUTE))])
async def upload_font(file: UploadFile = File(...)):
    safe_name = Path(file.filename or "custom_font.ttf").name
    if Path(safe_name).suffix.lower() not in {".ttf", ".otf"}:
        raise HTTPException(status_code=400, detail="Only .ttf and .otf font files are supported.")
    raw = await _read_upload_limited(file, 15 * 1024 * 1024, label="Font file")
    try:
        ImageFont.truetype(io.BytesIO(raw), size=16)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted font file.")
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = FONTS_DIR / safe_name
    dest.write_bytes(raw)
    return {"font": safe_name, "bytes": len(raw)}


async def _build_render_options(
    font_name: Optional[str],
    font_file: Optional[UploadFile],
    font_scale: float,
    text_color: str,
    outline_enabled: bool,
) -> RenderOptions:
    font_path: Optional[Path] = None
    font_bytes: Optional[bytes] = None
    if font_file is not None and font_file.filename:
        raw = await _read_upload_limited(font_file, 15 * 1024 * 1024, label="Font file")
        try:
            ImageFont.truetype(io.BytesIO(raw), size=16)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid or corrupted font_file.")
        font_bytes = raw
    elif font_name:
        safe = Path(font_name).name
        candidate = FONTS_DIR / safe
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"Font '{safe}' not found in fonts/.")
        font_path = candidate

    return RenderOptions(
        font_path=font_path,
        font_bytes=font_bytes,
        font_scale=max(0.4, min(2.5, float(font_scale))),
        text_color=text_color,
        outline_enabled=outline_enabled,
    )


def _encode_image(img_bgr: np.ndarray, output_format: str = "auto", quality: int = 94) -> str:
    fmt = output_format.strip().lower()
    if fmt == "auto":
        fmt = "jpeg" if img_bgr.shape[0] > 4000 else "png"
    q = max(50, min(100, int(quality)))
    if fmt in {"jpg", "jpeg"}:
        ok, encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        mime = "image/jpeg"
    elif fmt == "webp":
        ok, encoded = cv2.imencode(".webp", img_bgr, [int(cv2.IMWRITE_WEBP_QUALITY), q])
        mime = "image/webp"
    elif fmt == "png":
        ok, encoded = cv2.imencode(".png", img_bgr)
        mime = "image/png"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported output_format: {output_format}")
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")
    return f"data:{mime};base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


class ProviderConfigReq(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    selected_model: Optional[str] = None


class ProviderModelsReq(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@app.get("/api/providers")
async def list_providers():
    return {"providers": get_all_public_providers()}


@app.post("/api/providers/{provider_id}/config")
async def update_provider_config(provider_id: str, body: ProviderConfigReq):
    try:
        info = save_provider_secret(
            provider_id,
            api_key=body.api_key,
            base_url=body.base_url,
            selected_model=body.selected_model,
        )
        return {"provider": info}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/providers/{provider_id}/key")
async def remove_provider_key(provider_id: str):
    try:
        info = delete_provider_key(provider_id)
        return {"provider": info}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/providers/{provider_id}/models", dependencies=[Depends(_rate_dep("models", MODEL_DISCOVERY_RATE_LIMIT_PER_MINUTE))])
async def discover_provider_models(provider_id: str, body: ProviderModelsReq):
    try:
        models = await fetch_provider_models(
            provider_id, api_key=body.api_key, base_url=body.base_url
        )
        return {"provider_id": provider_id, "models": models}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch models: {e}")


@app.post("/api/translate", dependencies=[Depends(_rate_dep("translate", TRANSLATE_RATE_LIMIT_PER_MINUTE))])
async def translate_image(
    file: UploadFile = File(...),
    target_lang: str = Form("ar"),
    source_lang: str = Form("auto"),
    provider: str = Form("google_free"),
    model: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    mode: str = Form("full"),
    ocr_model: str = Form("auto"),
    inpaint_mode: str = Form("auto"),
    font_name: Optional[str] = Form(None),
    font_file: Optional[UploadFile] = File(None),
    font_scale: float = Form(1.0),
    text_color: str = Form("auto"),
    outline_enabled: bool = Form(True),
    output_format: str = Form("auto"),
    output_quality: int = Form(94),
):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is still initializing.")

    safe_name = Path(file.filename or "image.png").name
    raw = await _read_upload_limited(file, MAX_UPLOAD_BYTES, label=f"File '{safe_name}'")
    img_bgr = await asyncio.to_thread(_validate_and_decode_image, raw, safe_name)
    del raw

    render_opts = await _build_render_options(
        font_name, font_file, font_scale, text_color, outline_enabled
    )

    try:
        async with _gpu_semaphore:
            result = await asyncio.wait_for(
                _pipeline.process(
                    img_bgr,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    mode=mode,
                    ocr_model=ocr_model,
                    inpaint_mode=inpaint_mode,
                    render_options=render_opts,
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Translation request timed out.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Translation failed: {type(e).__name__}: {e}")

    b64_data_uri: Optional[str] = (
        None if mode == "ocr" else _encode_image(result.image_bgr, output_format, output_quality)
    )

    blocks_meta = [
        {
            "id": b.id,
            "source_text": b.text,
            "translated_text": b.translated_text,
            "bbox": list(b.bbox),
            "is_vertical": b.is_vertical,
            "is_sfx": b.is_sfx,
            "angle_deg": round(b.angle_deg, 1),
            "confidence": round(b.confidence, 3),
        }
        for b in result.blocks
    ]

    return JSONResponse(
        {
            "mode": mode,
            "image_base64": b64_data_uri,
            "width": int(result.image_bgr.shape[1]),
            "height": int(result.image_bgr.shape[0]),
            "target_lang": target_lang,
            "source_lang": source_lang,
            "device": result.device,
            "timings_ms": result.timings_ms,
            "timings_s": {k: round(v / 1000.0, 3) for k, v in result.timings_ms.items()},
            "inpaint_stats": result.inpaint_stats,
            "block_count": len(blocks_meta),
            "blocks": blocks_meta,
        }
    )


@app.post("/api/translate-batch", dependencies=[Depends(_rate_dep("translate", TRANSLATE_RATE_LIMIT_PER_MINUTE))])
async def translate_batch(
    files: List[UploadFile] = File(...),
    target_lang: str = Form("ar"),
    source_lang: str = Form("auto"),
    provider: str = Form("google_free"),
    model: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    consolidated: bool = Form(True),
    mode: str = Form("full"),
    ocr_model: str = Form("auto"),
    inpaint_mode: str = Form("auto"),
    font_name: Optional[str] = Form(None),
    font_file: Optional[UploadFile] = File(None),
    font_scale: float = Form(1.0),
    text_color: str = Form("auto"),
    outline_enabled: bool = Form(True),
    output_format: str = Form("jpeg"),
    output_quality: int = Form(95),
):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is still initializing.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Batch exceeds maximum allowed file count ({MAX_BATCH_FILES}).",
        )

    raw_items: List[Tuple[str, bytes]] = []
    total_batch_bytes = 0
    for idx, f in enumerate(files):
        safe_name = Path(f.filename or f"page_{idx + 1}.jpg").name
        raw = await _read_upload_limited(f, MAX_UPLOAD_BYTES, label=f"File '{safe_name}'")
        total_batch_bytes += len(raw)
        if total_batch_bytes > MAX_BATCH_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Batch aggregate payload exceeds maximum allowed size ({MAX_BATCH_TOTAL_BYTES // (1024 * 1024)} MB).",
            )
        raw_items.append((safe_name, raw))

    render_opts = await _build_render_options(
        font_name, font_file, font_scale, text_color, outline_enabled
    )

    def _encode_one(res) -> Optional[str]:
        if mode == "ocr":
            return None
        return _encode_image(res.image_bgr, output_format, output_quality)

    def _encode_chunk(chunk_results) -> List[Optional[str]]:
        if mode == "ocr":
            return [None] * len(chunk_results)
        with ThreadPoolExecutor(max_workers=4) as ex:
            return list(ex.map(_encode_one, chunk_results))

    # 2-stage pipelined chapter execution in memory-bounded 8-page chunks:
    # GPU OCR+Inpaint for chunk N+1 runs in parallel while chunk N awaits network translation & renders.
    CHAPTER_CHUNK_SIZE = 8
    in_flight_sem = asyncio.Semaphore(2)
    chunk_compute_lock = asyncio.Lock()
    t_batch_start = time.perf_counter()

    async def _process_one_chunk(c_start: int):
        async with in_flight_sem:
            c_end = min(len(raw_items), c_start + CHAPTER_CHUNK_SIZE)
            chunk_names = [raw_items[idx][0] for idx in range(c_start, c_end)]
            imgs_bgr = [
                await asyncio.to_thread(_validate_and_decode_image, raw_items[idx][1], raw_items[idx][0])
                for idx in range(c_start, c_end)
            ]
            for idx_clear in range(c_start, c_end):
                raw_items[idx_clear] = (raw_items[idx_clear][0], b"")

            trans_task: Optional[asyncio.Task] = None
            state = None
            try:
                async with _gpu_semaphore, chunk_compute_lock:
                    slice_imgs, state = await _pipeline.run_chunk_ocr(
                        imgs_bgr,
                        consolidated=consolidated,
                        mode=mode,
                        ocr_model=ocr_model,
                    )
                    del imgs_bgr
                    trans_task = asyncio.create_task(
                        _pipeline.translate_chunk_state(
                            state,
                            target_lang=target_lang,
                            source_lang=source_lang,
                            provider=provider,
                            model=model,
                            api_key=api_key,
                            base_url=base_url,
                            consolidated=consolidated,
                            mode=mode,
                        )
                    )
                    await _pipeline.run_chunk_inpaint(
                        slice_imgs, state, mode=mode, inpaint_mode=inpaint_mode
                    )
                    del slice_imgs

                translated_flat, t_trans = await trans_task
                chunk_results, chunk_timings = await _pipeline.finalize_chapter_chunk(
                    state,
                    translated_flat,
                    t_trans,
                    target_lang=target_lang,
                    mode=mode,
                    render_options=render_opts,
                )
            except BaseException:
                if trans_task is not None and not trans_task.done():
                    trans_task.cancel()
                    try:
                        await trans_task
                    except BaseException:
                        pass
                raise

            b64_list = await asyncio.to_thread(_encode_chunk, chunk_results)
            chunk_items = []
            for safe_name, res, b64_img in zip(chunk_names, chunk_results, b64_list):
                blocks_meta = [
                    {
                        "id": b.id,
                        "source_text": b.text,
                        "translated_text": b.translated_text,
                        "bbox": list(b.bbox),
                        "is_vertical": b.is_vertical,
                        "is_sfx": b.is_sfx,
                        "angle_deg": round(b.angle_deg, 1),
                        "confidence": round(b.confidence, 3),
                    }
                    for b in res.blocks
                ]
                chunk_items.append(
                    {
                        "filename": safe_name,
                        "image_base64": b64_img,
                        "width": int(res.image_bgr.shape[1]),
                        "height": int(res.image_bgr.shape[0]),
                        "timings_ms": res.timings_ms,
                        "inpaint_stats": res.inpaint_stats,
                        "block_count": len(blocks_meta),
                        "blocks": blocks_meta,
                    }
                )
            del chunk_results, b64_list, state
            gc.collect()
            return chunk_items, chunk_timings

    try:
        chunk_starts = range(0, len(raw_items), CHAPTER_CHUNK_SIZE)
        chunk_outputs = await asyncio.wait_for(
            asyncio.gather(*(_process_one_chunk(cs) for cs in chunk_starts)),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Batch translation request timed out.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Batch translation failed: {type(e).__name__}: {e}")

    items = []
    chapter_timings: Dict[str, float] = {
        "ocr": 0.0,
        "translate": 0.0,
        "inpaint": 0.0,
        "render": 0.0,
        "total": 0.0,
        "slices_processed": 0,
    }
    for chunk_items, chunk_timings in chunk_outputs:
        items.extend(chunk_items)
        for k in ("ocr", "translate", "inpaint", "render", "slices_processed"):
            chapter_timings[k] += chunk_timings.get(k, 0)

    wall_total_ms = (time.perf_counter() - t_batch_start) * 1000.0
    for k in ("ocr", "translate", "inpaint", "render"):
        chapter_timings[k] = round(chapter_timings[k], 1)
    chapter_timings["total"] = round(wall_total_ms, 1)
    chapter_timings["per_image_avg"] = round(wall_total_ms / max(1, len(items)), 1)

    return JSONResponse(
        {
            "mode": mode,
            "image_count": len(items),
            "target_lang": target_lang,
            "source_lang": source_lang,
            "device": _pipeline.active_device,
            "chapter_timings_ms": chapter_timings,
            "chapter_timings_s": {
                k: round(v / 1000.0, 3) if k != "slices_processed" else v
                for k, v in chapter_timings.items()
            },
            "consolidated": consolidated,
            "results": items,
        }
    )


@app.post("/api/ocr", dependencies=[Depends(_rate_dep("translate", TRANSLATE_RATE_LIMIT_PER_MINUTE))])
async def ocr_only(
    files: List[UploadFile] = File(...),
    consolidated: bool = Form(True),
    ocr_model: str = Form("auto"),
):
    """Extract text blocks only (skips translation, inpainting, and typesetting)."""
    return await translate_batch(
        files=files,
        target_lang="ar",
        source_lang="auto",
        provider="google_free",
        model=None,
        api_key=None,
        base_url=None,
        consolidated=consolidated,
        mode="ocr",
        ocr_model=ocr_model,
        inpaint_mode="auto",
        font_name=None,
        font_file=None,
        font_scale=1.0,
        text_color="auto",
        outline_enabled=True,
        output_format="jpeg",
        output_quality=95,
    )
