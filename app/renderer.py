"""Manga-style text renderer with dynamic contrast, Arabic/RTL shaping, and angled/curved layout support."""

from __future__ import annotations

import hashlib
import io
import math
import struct
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from app.config import (
    FONT_ARABIC,
    FONT_ARABIC_SFX,
    FONT_FALLBACK,
    FONT_LATIN,
    FONT_SFX,
    RTL_LANGUAGES,
)
from app.ocr_engine import TextBlock

_HAS_RAQM = features.check("raqm")
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    arabic_reshaper = None
    get_display = None


@dataclass
class RenderOptions:
    """Optional per-request rendering customization."""

    font_path: Optional[Path] = None
    font_bytes: Optional[bytes] = None
    font_scale: float = 1.0
    text_color: str = "auto"  # "auto" or "#RRGGBB"
    outline_enabled: bool = True


def _parse_hex_color(color_str: str) -> Optional[Tuple[int, int, int]]:
    s = color_str.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return None


_FONT_CMAP_CACHE: dict[str, set[int]] = {}


def _read_ttf_cmap(data: bytes) -> set[int]:
    """Parse TrueType/OpenType cmap subtables (format 4 & 12) to list supported codepoints."""
    try:
        num_tables = struct.unpack(">H", data[4:6])[0]
        for i in range(num_tables):
            off = 12 + i * 16
            tag, _, tbl_off, tbl_len = struct.unpack(">4sIII", data[off : off + 16])
            if tag == b"cmap":
                tbl = data[tbl_off : tbl_off + tbl_len]
                nsub = struct.unpack(">H", tbl[2:4])[0]
                cps: set[int] = set()
                for j in range(nsub):
                    _, _, sub_off = struct.unpack(">HHI", tbl[4 + j * 8 : 12 + j * 8])
                    fmt = struct.unpack(">H", tbl[sub_off : sub_off + 2])[0]
                    if fmt == 4:
                        seg_count = struct.unpack(">H", tbl[sub_off + 6 : sub_off + 8])[0] // 2
                        end_codes = struct.unpack(
                            f">{seg_count}H", tbl[sub_off + 14 : sub_off + 14 + 2 * seg_count]
                        )
                        start_codes = struct.unpack(
                            f">{seg_count}H",
                            tbl[sub_off + 16 + 2 * seg_count : sub_off + 16 + 4 * seg_count],
                        )
                        for s, e in zip(start_codes, end_codes):
                            if s != 0xFFFF:
                                cps.update(range(s, e + 1))
                return cps
    except Exception:
        pass
    return set()


def _get_font_cmap(font_path: Path, font_bytes: Optional[bytes] = None) -> set[int]:
    if font_bytes is not None:
        key = "bytes:" + hashlib.sha256(font_bytes).hexdigest()[:16]
        if key not in _FONT_CMAP_CACHE:
            _FONT_CMAP_CACHE[key] = _read_ttf_cmap(font_bytes)
        return _FONT_CMAP_CACHE[key]
    key = str(font_path)
    if key not in _FONT_CMAP_CACHE:
        try:
            _FONT_CMAP_CACHE[key] = _read_ttf_cmap(font_path.read_bytes())
        except Exception:
            _FONT_CMAP_CACHE[key] = set()
    return _FONT_CMAP_CACHE[key]


_CMAP_BY_ID: dict[int, set[int]] = {}


@lru_cache(maxsize=4096)
def _shape_rtl_cached(text: str, cmap_id: int) -> str:
    shaped = get_display(arabic_reshaper.reshape(text))
    if cmap_id:
        font_cmap = _CMAP_BY_ID.get(cmap_id)
        if font_cmap:
            shaped = "".join(
                ch if (ord(ch) in font_cmap or ch.isspace()) else unicodedata.normalize("NFKC", ch)
                for ch in shaped
            )
    return shaped


class MangaRenderer:
    """Renders translated text onto cleaned manga pages with adaptive contrast & manga fonts."""

    def __init__(self) -> None:
        self._font_cache: dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def _pick_font_path(self, target_lang: str, is_sfx: bool) -> Path:
        if target_lang in RTL_LANGUAGES:
            if is_sfx and FONT_ARABIC_SFX.exists():
                return FONT_ARABIC_SFX
            if FONT_ARABIC.exists():
                return FONT_ARABIC
        if target_lang in {"zh-CN", "zh-TW", "ja", "ko"}:
            cjk_candidates = (
                [Path("C:/Windows/Fonts/msyhbd.ttc"), Path("C:/Windows/Fonts/msyh.ttc")]
                if target_lang in {"zh-CN", "zh-TW"}
                else (
                    [Path("C:/Windows/Fonts/YuGothB.ttc"), Path("C:/Windows/Fonts/msyhbd.ttc")]
                    if target_lang == "ja"
                    else [Path("C:/Windows/Fonts/malgunbd.ttf"), Path("C:/Windows/Fonts/msyhbd.ttc")]
                )
            )
            cjk_candidates.extend([
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
                Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"),
                Path("/System/Library/Fonts/PingFang.ttc"),
            ])
            for cand in cjk_candidates:
                if cand.exists():
                    return cand
        if target_lang in {"ja", "ko", "zh-CN", "zh-TW", "th", "hi", "ru", "uk"}:
            if FONT_FALLBACK.exists():
                return FONT_FALLBACK
        if is_sfx and FONT_SFX.exists():
            return FONT_SFX
        if FONT_LATIN.exists():
            return FONT_LATIN
        if FONT_FALLBACK.exists():
            return FONT_FALLBACK
        # System fallback (Windows Arial / Segoe UI)
        win_arial = Path("C:/Windows/Fonts/arialbd.ttf")
        return win_arial if win_arial.exists() else Path("C:/Windows/Fonts/segoeui.ttf")

    def _get_font(
        self, font_path: Path, size: int, font_bytes: Optional[bytes] = None
    ) -> ImageFont.FreeTypeFont:
        if font_bytes is not None:
            digest = hashlib.sha256(font_bytes).hexdigest()[:16]
            key = (f"bytes:{digest}", size)
            if key not in self._font_cache:
                try:
                    layout_engine = ImageFont.Layout.RAQM if _HAS_RAQM else ImageFont.Layout.BASIC
                    self._font_cache[key] = ImageFont.truetype(
                        io.BytesIO(font_bytes), size=size, layout_engine=layout_engine
                    )
                except Exception:
                    self._font_cache[key] = ImageFont.truetype(io.BytesIO(font_bytes), size=size)
            return self._font_cache[key]

        key = (str(font_path), size)
        if key not in self._font_cache:
            try:
                layout_engine = ImageFont.Layout.RAQM if _HAS_RAQM else ImageFont.Layout.BASIC
                self._font_cache[key] = ImageFont.truetype(
                    str(font_path), size=size, layout_engine=layout_engine
                )
            except Exception:
                self._font_cache[key] = ImageFont.truetype(str(font_path), size=size)
        return self._font_cache[key]

    @staticmethod
    def _prepare_text(text: str, is_rtl: bool, font_cmap: Optional[set[int]] = None) -> str:
        if not is_rtl or _HAS_RAQM:
            return text
        if arabic_reshaper is not None and get_display is not None:
            cmap_id = 0
            if font_cmap:
                cmap_id = id(font_cmap)
                _CMAP_BY_ID[cmap_id] = font_cmap
            return _shape_rtl_cached(text, cmap_id)
        return text

    @staticmethod
    def _text_width(
        draw: ImageDraw.ImageDraw,
        disp: str,
        font: ImageFont.FreeTypeFont,
        is_rtl: bool,
        w_cache: Optional[dict[Tuple[int, str], int]] = None,
    ) -> int:
        if w_cache is not None:
            ck = (id(font), disp)
            cached = w_cache.get(ck)
            if cached is not None:
                return cached
        kwargs = {"font": font}
        if is_rtl and _HAS_RAQM:
            kwargs["direction"] = "rtl"
        tb = draw.textbbox((0, 0), disp, **kwargs)
        w = tb[2] - tb[0]
        if w_cache is not None:
            w_cache[(id(font), disp)] = w
        return w

    @staticmethod
    def _wrap_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
        is_rtl: bool,
        font_cmap: Optional[set[int]] = None,
        w_cache: Optional[dict[Tuple[int, str], int]] = None,
    ) -> Tuple[List[str], int]:
        has_cjk = any("\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" for ch in text)
        if has_cjk and len(text.split()) <= 2:
            lines: List[str] = []
            current = ""
            cur_w = 0
            max_w = 0
            for ch in text:
                candidate = current + ch
                cw = MangaRenderer._text_width(draw, candidate, font, is_rtl, w_cache)
                if cw <= max_width or not current:
                    current = candidate
                    cur_w = cw
                else:
                    lines.append(current)
                    max_w = max(max_w, cur_w)
                    current = ch
                    cur_w = MangaRenderer._text_width(draw, current, font, is_rtl, w_cache)
            if current:
                lines.append(current)
                max_w = max(max_w, cur_w)
            return lines, max_w

        words = text.split()
        if not words:
            disp = MangaRenderer._prepare_text(text, is_rtl, font_cmap)
            return [text], MangaRenderer._text_width(draw, disp, font, is_rtl, w_cache)
        lines: List[str] = []
        current = ""
        cur_w = 0
        max_w = 0
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            disp = MangaRenderer._prepare_text(candidate, is_rtl, font_cmap)
            cw = MangaRenderer._text_width(draw, disp, font, is_rtl, w_cache)
            if cw <= max_width or not current:
                current = candidate
                cur_w = cw
            else:
                lines.append(current)
                max_w = max(max_w, cur_w)
                current = word
                w_disp = MangaRenderer._prepare_text(word, is_rtl, font_cmap)
                cur_w = MangaRenderer._text_width(draw, w_disp, font, is_rtl, w_cache)
        if current:
            lines.append(current)
            max_w = max(max_w, cur_w)
        return lines, max_w

    def _fit_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font_path: Path,
        box_w: int,
        box_h: int,
        is_rtl: bool,
        max_size: int = 44,
        min_size: int = 8,
        font_bytes: Optional[bytes] = None,
        font_cmap: Optional[set[int]] = None,
    ) -> Tuple[ImageFont.FreeTypeFont, List[str], int, int]:
        """Find largest font size where wrapped text fits comfortably inside (box_w, box_h)."""
        pad_w = max(6, int(box_w * 0.88))
        pad_h = max(6, int(box_h * 0.88))
        w_cache: dict[Tuple[int, str], int] = {}

        lo, hi = min_size, max(min_size, min(max_size, int(min(box_w, box_h) * 0.72)))
        best_font = self._get_font(font_path, lo, font_bytes=font_bytes)
        best_lines, _ = self._wrap_text(draw, text, best_font, pad_w, is_rtl, font_cmap, w_cache)
        best_line_h = int(lo * 1.25)

        while lo <= hi:
            mid = (lo + hi) // 2
            font = self._get_font(font_path, mid, font_bytes=font_bytes)
            lines, max_line_w = self._wrap_text(draw, text, font, pad_w, is_rtl, font_cmap, w_cache)
            line_h = int(mid * 1.25)
            total_h = line_h * len(lines)
            if total_h <= pad_h and max_line_w <= pad_w:
                best_font = font
                best_lines = lines
                best_line_h = line_h
                lo = mid + 1
            else:
                hi = mid - 1

        total_h = best_line_h * len(best_lines)
        return best_font, best_lines, best_line_h, total_h

    @staticmethod
    def _analyze_background(
        cleaned_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        gray_img: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool]:
        """
        Inspect cleaned region luminance & variance.
        Returns (fill_rgb, stroke_rgb, needs_outline).
        - White text on dark backgrounds, black text on light backgrounds.
        """
        h, w = cleaned_bgr.shape[:2]
        x0, y0, x1, y1 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
        if x1 <= x0 or y1 <= y0:
            return (17, 17, 17), (255, 255, 255), False

        gray = (
            gray_img[y0:y1, x0:x1]
            if gray_img is not None
            else cv2.cvtColor(cleaned_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        )
        lum = float(np.median(gray))
        std = float(np.std(gray))
        needs_outline = std > 18.0

        if lum < 128.0:
            return (255, 255, 255), (15, 15, 15), needs_outline
        return (17, 17, 17), (255, 255, 255), needs_outline

    def render(
        self,
        cleaned_bgr: np.ndarray,
        blocks: List[TextBlock],
        target_lang: str,
        options: Optional[RenderOptions] = None,
    ) -> np.ndarray:
        opts = options or RenderOptions()
        custom_rgb = _parse_hex_color(opts.text_color) if opts.text_color != "auto" else None
        is_rtl = target_lang in RTL_LANGUAGES
        gray_full = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for blk in blocks:
            txt = (blk.translated_text or blk.text or "").strip()
            if not txt:
                continue

            x0, y0, x1, y1 = blk.bbox
            bw, bh = max(24, x1 - x0), max(20, y1 - y0)

            fill_rgb, stroke_rgb, needs_outline = self._analyze_background(
                cleaned_bgr, (x0, y0, x1, y1), gray_img=gray_full
            )
            if custom_rgb is not None:
                fill_rgb = custom_rgb
                lum = 0.299 * fill_rgb[0] + 0.587 * fill_rgb[1] + 0.114 * fill_rgb[2]
                stroke_rgb = (15, 15, 15) if lum >= 128 else (255, 255, 255)
            if not opts.outline_enabled:
                needs_outline = False

            font_path = opts.font_path if (opts.font_path and opts.font_path.exists()) else self._pick_font_path(target_lang, blk.is_sfx)
            font_bytes = opts.font_bytes
            font_cmap = _get_font_cmap(font_path, font_bytes) if (is_rtl and not _HAS_RAQM) else None

            # Only rotate true full-height thin margin strips (e.g. copyright) or consistent slanted text
            tall_vertical = bw < 32 and bh > bw * 6.0 and len(txt) > 15
            ang_std = float(np.std([l.angle_deg for l in blk.lines])) if blk.lines else 0.0
            angled = (not blk.is_vertical) and (8.0 < abs(blk.angle_deg) < 72.0) and ang_std < 12.0

            # Safely expand narrow bubbles ONLY within uniform bubble interior (works on white, black & colored bubbles)
            narrow_thresh = max(96, int(pil_img.width * 0.18))
            max_expand = max(24, int(pil_img.width * 0.045))
            margin_pad = max(3, int(pil_img.width * 0.005))
            if (not tall_vertical) and bw < narrow_thresh and bh > bw * 1.35:
                cy = min(cleaned_bgr.shape[0] - 1, max(0, (y0 + y1) // 2))
                cx = min(cleaned_bgr.shape[1] - 1, max(0, (x0 + x1) // 2))
                patch_g = gray_full[max(0, cy - 3) : min(gray_full.shape[0], cy + 4), max(0, cx - 3) : min(gray_full.shape[1], cx + 4)]
                if patch_g.size > 0 and float(np.std(patch_g)) < 15.0:
                    center_bgr = cleaned_bgr[cy, cx].astype(np.int16)
                    lx_lim = x0
                    while lx_lim > max(0, x0 - max_expand) and int(np.abs(cleaned_bgr[cy, lx_lim - 1].astype(np.int16) - center_bgr).sum()) <= 45:
                        lx_lim -= 1
                    rx_lim = x1
                    while rx_lim < min(pil_img.width, x1 + max_expand) and int(np.abs(cleaned_bgr[cy, rx_lim].astype(np.int16) - center_bgr).sum()) <= 45:
                        rx_lim += 1
                    if (rx_lim - lx_lim) > bw + 2 * margin_pad:
                        x0, x1 = lx_lim + margin_pad, rx_lim - margin_pad
                        bw = max(20, x1 - x0)

            if tall_vertical or angled:
                if pil_img.mode != "RGBA":
                    pil_img = pil_img.convert("RGBA")
                    draw = ImageDraw.Draw(pil_img)
                render_angle = 90.0 if tall_vertical else blk.angle_deg
                fit_w, fit_h = (bh, bw) if tall_vertical else (bw, bh)
                diag = int(math.hypot(bw, bh)) + 20
                layer = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
                ldraw = ImageDraw.Draw(layer)
                font, lines, line_h, total_h = self._fit_text(
                    ldraw,
                    txt,
                    font_path,
                    fit_w,
                    fit_h,
                    is_rtl,
                    max_size=max(10, int(max(24, blk.avg_font_size * 1.1) * opts.font_scale)),
                    font_bytes=font_bytes,
                    font_cmap=font_cmap,
                )
                cur_y = (diag - total_h) // 2
                stroke_w = max(2, font.size // 10) if needs_outline else 0
                for ln in lines:
                    disp = self._prepare_text(ln, is_rtl, font_cmap)
                    lw = self._text_width(ldraw, disp, font, is_rtl)
                    lx = (diag - lw) // 2
                    kwargs = {
                        "font": font,
                        "fill": (*fill_rgb, 255),
                        "stroke_width": stroke_w,
                    }
                    if stroke_w:
                        kwargs["stroke_fill"] = (*stroke_rgb, 255)
                    if is_rtl and _HAS_RAQM:
                        kwargs["direction"] = "rtl"
                    ldraw.text((lx, cur_y), disp, **kwargs)
                    cur_y += line_h
                rotated = layer.rotate(-render_angle, resample=Image.Resampling.BICUBIC, expand=False)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                pil_img.alpha_composite(rotated, (cx - diag // 2, cy - diag // 2))
            else:
                font, lines, line_h, total_h = self._fit_text(
                    draw,
                    txt,
                    font_path,
                    bw,
                    bh,
                    is_rtl,
                    max_size=max(10, int(max(28, blk.avg_font_size * 1.15) * opts.font_scale)),
                    font_bytes=font_bytes,
                    font_cmap=font_cmap,
                )
                cur_y = y0 + max(0, (bh - total_h) // 2)
                stroke_w = max(2, font.size // 10) if needs_outline else 0
                for ln in lines:
                    disp = self._prepare_text(ln, is_rtl, font_cmap)
                    lw = self._text_width(draw, disp, font, is_rtl)
                    lx = x0 + max(0, (bw - lw) // 2)
                    kwargs = {
                        "font": font,
                        "fill": fill_rgb if pil_img.mode == "RGB" else (*fill_rgb, 255),
                        "stroke_width": stroke_w,
                    }
                    if stroke_w:
                        kwargs["stroke_fill"] = stroke_rgb if pil_img.mode == "RGB" else (*stroke_rgb, 255)
                    if is_rtl and _HAS_RAQM:
                        kwargs["direction"] = "rtl"
                    draw.text((lx, cur_y), disp, **kwargs)
                    cur_y += line_h

        out_rgb = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
        return cv2.cvtColor(np.array(out_rgb), cv2.COLOR_RGB2BGR)
