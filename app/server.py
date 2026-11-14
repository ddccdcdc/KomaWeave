"""FastAPI server for the Manga & Manhwa Translation System."""

from __future__ import annotations

import asyncio
import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import ImageFont
from pydantic import BaseModel

from app.config import FONTS_DIR, LANGUAGES, STATIC_DIR
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
_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
_MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "300"))

app = FastAPI(title="KomaWeave", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/api/health")
async def health():
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif mps_avail:
        device_name = "Apple MPS"
    else:
        device_name = "CPU"
    ocr_loaded = _pipeline is not None and _pipeline._ocr is not None
    aot_loaded = (
        _pipeline is not None
        and _pipeline._inpainter is not None
        and _pipeline._inpainter._model is not None
    )
    return {
        "status": "ready" if _pipeline is not None else "initializing",
        "version": app.version,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps_avail,
        "ocr_loaded": ocr_loaded,
        "aot_loaded": aot_loaded,
        "ocr_cuda": bool(_pipeline._ocr.use_cuda) if ocr_loaded else False,
        "device": device_name,
    }


@app.get("/api/fonts")
async def list_fonts():
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    fonts = sorted(
        p.name for p in FONTS_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".ttf", ".otf"}
    )
    return {"fonts": fonts}


@app.post("/api/fonts")
async def upload_font(file: UploadFile = File(...)):
    safe_name = Path(file.filename or "custom_font.ttf").name
    if Path(safe_name).suffix.lower() not in {".ttf", ".otf"}:
        raise HTTPException(status_code=400, detail="Only .ttf and .otf font files are supported.")
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Font file exceeds 15 MB limit.")
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
        raw = await font_file.read()
        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Font file exceeds 15 MB limit.")
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


@app.post("/api/providers/{provider_id}/models")
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


@app.post("/api/translate")
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

    raw = await file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size ({_MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image file.")

    render_opts = await _build_render_options(
        font_name, font_file, font_scale, text_color, outline_enabled
    )

    try:
        result = await _pipeline.process(
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
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Translation failed: {e}")

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


@app.post("/api/translate-batch")
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
    if len(files) > _MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Batch exceeds maximum allowed file count ({_MAX_BATCH_FILES}).",
        )

    raw_bytes_list = await asyncio.gather(*(f.read() for f in files))
    for idx, raw in enumerate(raw_bytes_list):
        if len(raw) > _MAX_UPLOAD_BYTES:
            safe_name = Path(files[idx].filename or f"page_{idx + 1}.jpg").name
            raise HTTPException(
                status_code=413,
                detail=f"File '{safe_name}' exceeds maximum allowed size ({_MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
            )

    def _decode_all() -> List[np.ndarray]:
        imgs: List[np.ndarray] = []
        for idx, raw in enumerate(raw_bytes_list):
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid or unsupported image file: {files[idx].filename}",
                )
            imgs.append(img)
        return imgs

    render_opts = await _build_render_options(
        font_name, font_file, font_scale, text_color, outline_enabled
    )

    imgs_bgr = await asyncio.to_thread(_decode_all)
    try:
        results, chapter_timings = await _pipeline.process_chapter(
            imgs_bgr,
            target_lang=target_lang,
            source_lang=source_lang,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            consolidated=consolidated,
            mode=mode,
            ocr_model=ocr_model,
            inpaint_mode=inpaint_mode,
            render_options=render_opts,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Batch translation failed: {e}")

    def _encode_one(res) -> Optional[str]:
        if mode == "ocr":
            return None
        return _encode_image(res.image_bgr, output_format, output_quality)

    def _encode_all() -> List[Optional[str]]:
        if mode == "ocr":
            return [None] * len(results)
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(_encode_one, results))

    b64_list = await asyncio.to_thread(_encode_all)

    items = []
    for f, res, b64_img in zip(files, results, b64_list):
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
        items.append(
            {
                "filename": Path(f.filename or f"page_{len(items) + 1}.jpg").name,
                "image_base64": b64_img,
                "width": int(res.image_bgr.shape[1]),
                "height": int(res.image_bgr.shape[0]),
                "timings_ms": res.timings_ms,
                "inpaint_stats": res.inpaint_stats,
                "block_count": len(blocks_meta),
                "blocks": blocks_meta,
            }
        )

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


@app.post("/api/ocr")
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
