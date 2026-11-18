"""General-purpose multilingual OCR & layout-aware clustering (RapidOCR PP-OCRv6)."""

from __future__ import annotations

import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
import torch

# Ensure PyTorch's CUDA 12 DLLs are visible to ONNX Runtime CUDA EP on Windows
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(_torch_lib):
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_torch_lib)
        except OSError:
            pass
    if _torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

from rapidocr import ModelType, RapidOCR
from rapidocr.ch_ppocr_rec.typings import TextRecInput

from app.config import (
    OCR_DET_BOX_THRESH,
    OCR_DET_LIMIT_SIDE,
    OCR_DET_THRESH,
    OCR_DET_UNCLIP_RATIO,
)

_KANA_ONLY_RE = re.compile(r"^[\u3040-\u309F\u30A0-\u30FFー・\s]+$")
_NOISE_RE = re.compile(r"^[\W_0-9%.,:;/\\()\[\]!?+\-]{1,12}$")
_NUM_GARBAGE_RE = re.compile(r"^[\d\s.,:%/()\[\]\-#*+~_=]+$")
_PUNCT_ONLY_RE = re.compile(r"^[!?！？…・。、.~—–\-]+$")


@dataclass
class TextLine:
    poly: np.ndarray  # (4, 2) float32
    text: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float
    cx: float
    cy: float
    w: float
    h: float
    is_vertical: bool
    font_size: float
    angle_deg: float
    is_furigana: bool = False


@dataclass
class TextBlock:
    id: int
    lines: List[TextLine] = field(default_factory=list)
    text: str = ""
    translated_text: str = ""
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    polygon: np.ndarray | None = None
    inpaint_polygons: List[np.ndarray] = field(default_factory=list)
    is_vertical: bool = False
    is_sfx: bool = False
    angle_deg: float = 0.0
    avg_font_size: float = 16.0
    confidence: float = 0.0


class GeneralOCREngine:
    """High-speed, general-purpose multilingual OCR using RapidOCR PP-OCRv6."""

    def __init__(self, use_cuda: bool | None = None, use_dml: bool | None = None) -> None:
        avail_providers: list[str] = []
        try:
            import onnxruntime as ort

            avail_providers = list(ort.get_available_providers())
        except Exception:
            pass

        if use_cuda is None:
            use_cuda = torch.cuda.is_available() and ("CUDAExecutionProvider" in avail_providers)
        self.use_cuda = bool(use_cuda)

        if use_dml is None:
            use_dml = (not self.use_cuda) and ("DmlExecutionProvider" in avail_providers)
        self.use_dml = bool(use_dml)

        self._lock = threading.Lock()
        try:
            self.det_engine = self._build_det_engine(
                use_cuda=self.use_cuda, use_dml=self.use_dml, fast_mode=False
            )
        except Exception:
            if not (self.use_cuda or self.use_dml):
                raise
            self.use_cuda = False
            self.use_dml = False
            self.det_engine = self._build_det_engine(use_cuda=False, use_dml=False, fast_mode=False)
        self._fast_det_engine: RapidOCR | None = None
        self.rec_engine = self._build_rec_engine(fast_mode=False)

    @property
    def fast_det_engine(self) -> RapidOCR:
        if self._fast_det_engine is None:
            with self._lock:
                if self._fast_det_engine is None:
                    self._fast_det_engine = self._build_det_engine(
                        use_cuda=self.use_cuda, use_dml=self.use_dml, fast_mode=True
                    )
        return self._fast_det_engine

    @staticmethod
    def _ensure_det_td(eng: RapidOCR):
        if eng.text_det is None:
            loader = getattr(eng, "_load_det_model", None)
            if callable(loader):
                loader()
            if eng.text_det is None:
                from rapidocr.ch_ppocr_det import TextDetector

                cfg = eng.cfg
                cfg.Det.engine_cfg = cfg.EngineConfig[cfg.Det.engine_type.value]
                cfg.Det.model_root_dir = cfg.Global.model_root_dir
                eng.text_det = TextDetector(cfg.Det)
        return eng.text_det

    @staticmethod
    def _ensure_rec_tr(eng: RapidOCR):
        if eng.text_rec is None:
            loader = getattr(eng, "_load_rec_model", None)
            if callable(loader):
                loader()
            if eng.text_rec is None:
                from rapidocr.ch_ppocr_rec import TextRecognizer

                cfg = eng.cfg
                cfg.Rec.engine_cfg = cfg.EngineConfig[cfg.Rec.engine_type.value]
                cfg.Rec.font_path = cfg.Global.font_path
                cfg.Rec.model_root_dir = cfg.Global.model_root_dir
                eng.text_rec = TextRecognizer(cfg.Rec)
        return eng.text_rec

    @classmethod
    def _build_det_engine(cls, use_cuda: bool, use_dml: bool = False, fast_mode: bool = False) -> RapidOCR:
        side = 800 if fast_mode else OCR_DET_LIMIT_SIDE
        params = {
            "EngineConfig.onnxruntime.use_cuda": use_cuda,
            "EngineConfig.onnxruntime.use_dml": use_dml,
            "Global.use_det": True,
            "Global.use_cls": False,
            "Global.use_rec": fast_mode,
            "Det.model_type": ModelType.TINY if fast_mode else ModelType.MEDIUM,
            "Rec.model_type": ModelType.TINY if fast_mode else ModelType.SMALL,
            "Det.limit_type": "max",
            "Det.limit_side_len": side,
            "Det.box_thresh": OCR_DET_BOX_THRESH,
            "Det.thresh": OCR_DET_THRESH,
            "Det.unclip_ratio": OCR_DET_UNCLIP_RATIO,
        }
        if use_cuda:
            params.update(
                {
                    "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "HEURISTIC",
                    "EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy": "kSameAsRequested",
                    "EngineConfig.onnxruntime.cuda_ep_cfg.gpu_mem_limit": 3 * 1024 * 1024 * 1024,
                }
            )
        eng = RapidOCR(params=params)
        cls._ensure_det_td(eng)
        if fast_mode:
            cls._ensure_rec_tr(eng)
        return eng

    @classmethod
    def _build_rec_engine(cls, fast_mode: bool = False, intra_threads: int = 0) -> RapidOCR:
        params = {
            "EngineConfig.onnxruntime.use_cuda": False,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Global.use_det": False,
            "Global.use_cls": False,
            "Global.use_rec": True,
            "Det.model_type": ModelType.TINY,
            "Rec.model_type": ModelType.TINY if fast_mode else ModelType.SMALL,
            "Rec.rec_batch_num": 12,
        }
        if intra_threads > 0:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = intra_threads
        eng = RapidOCR(params=params)
        cls._ensure_rec_tr(eng)
        return eng

    def ensure_fast_pool(self, workers: int = 4, fast_mode: bool = True) -> List[RapidOCR]:
        """Lazily initialize a thread-safe pool of CPU Rec workers with bounded ORT threads."""
        cpus = os.cpu_count() or 4
        target_workers = min(workers, max(2, cpus // 4))
        intra = max(2, cpus // target_workers)
        attr = "_fast_rec_pool" if fast_mode else "_std_rec_pool"
        pool = getattr(self, attr, None)
        if pool is None:
            pool = []
            setattr(self, attr, pool)
        while len(pool) < target_workers:
            pool.append(self._build_rec_engine(fast_mode=fast_mode, intra_threads=intra))
        return pool[:target_workers]

    @staticmethod
    def _poly_angle(poly: np.ndarray) -> float:
        """Compute horizontal baseline tilt angle in degrees (-90 to 90) from top edge."""
        p0, p1 = poly[0], poly[1]
        e_top = p1 - p0
        ang = math.degrees(math.atan2(float(e_top[1]), float(e_top[0])))
        while ang > 90:
            ang -= 180
        while ang < -90:
            ang += 180
        return ang

    @staticmethod
    def _crop_poly(img_bgr: np.ndarray, poly: np.ndarray, rotate_vertical: bool = True) -> np.ndarray:
        pts = poly.astype(np.float32)
        w = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
        h = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
        w, h = max(w, 4), max(h, 4)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        crop = cv2.warpPerspective(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if rotate_vertical and h >= w * 2.3:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return crop

    def _retry_rotated_crops(
        self, img_bgr: np.ndarray, lines: List[TextLine], engine: RapidOCR | None = None
    ) -> None:
        """Recover orientation for ambiguous aspect-ratio crops or upside-down/sideways lines."""
        candidates: List[Tuple[int, np.ndarray, bool]] = []  # (idx, crop, is_vertical_candidate)
        for idx, ln in enumerate(lines):
            aspect = ln.h / max(1.0, ln.w)
            if 1.35 <= aspect < 2.3 and (ln.score < 0.78 or len(ln.text) <= 1):
                if (not ln.text.strip()) or ln.font_size > 48:
                    continue
                raw_crop = self._crop_poly(img_bgr, ln.poly, rotate_vertical=False)
                candidates.append((idx, cv2.rotate(raw_crop, cv2.ROTATE_90_COUNTERCLOCKWISE), True))
            elif aspect >= 2.3 and (ln.score < 0.72 or (ln.text.isascii() and len(ln.text) <= 2)):
                # Test 180-deg flip (upside-down vertical) AND upright (for narrow ASCII 'I'/'!')
                crop_vert = self._crop_poly(img_bgr, ln.poly, rotate_vertical=True)
                candidates.append((idx, cv2.rotate(crop_vert, cv2.ROTATE_180), True))
                if ln.text.isascii() or len(ln.text) <= 2:
                    raw_crop = self._crop_poly(img_bgr, ln.poly, rotate_vertical=False)
                    candidates.append((idx, raw_crop, False))
            elif ln.score < 0.75 and (ln.is_vertical or abs(ln.angle_deg) > 15):
                crop = self._crop_poly(img_bgr, ln.poly)
                candidates.append((idx, cv2.rotate(crop, cv2.ROTATE_180), ln.is_vertical))

        if not candidates:
            return

        crops = [c[1] for c in candidates]
        rec_engine = engine or self.rec_engine
        rec_tr = self._ensure_rec_tr(rec_engine)
        rec_res = rec_tr(TextRecInput(img=crops))
        if not rec_res or not rec_res.txts:
            return

        best_by_idx: dict[int, Tuple[float, str, bool]] = {}
        for (idx, _, cand_vert), new_txt, new_score in zip(candidates, rec_res.txts, rec_res.scores):
            new_txt = (new_txt or "").strip()
            sc = float(new_score)
            if not new_txt:
                continue
            prev = best_by_idx.get(idx)
            if prev is None or sc > prev[0]:
                best_by_idx[idx] = (sc, new_txt, cand_vert)

        for idx, (best_sc, best_txt, cand_vert) in best_by_idx.items():
            if best_sc > lines[idx].score + 0.04:
                lines[idx].text = best_txt
                lines[idx].score = best_sc
                if not cand_vert and best_txt.isascii() and len(best_txt) <= 4:
                    lines[idx].is_vertical = False
                    lines[idx].font_size = lines[idx].h
                elif cand_vert:
                    lines[idx].is_vertical = True
                    lines[idx].font_size = lines[idx].w

    @staticmethod
    def _mark_furigana_and_noise(lines: List[TextLine], img_bgr: np.ndarray | None = None) -> List[TextLine]:
        """Flag tiny furigana ruby text beside main lines and drop pure graphic noise."""
        ascii_lines = sum(1 for l in lines if any(c.isascii() and c.isalpha() for c in l.text))
        mostly_ascii_page = ascii_lines >= 2 and ascii_lines >= 0.6 * max(1, len(lines))

        kept: List[TextLine] = []
        for i, a in enumerate(lines):
            txt = a.text.strip()
            if not txt:
                continue
            is_punct = bool(_PUNCT_ONLY_RE.match(txt))
            if _NUM_GARBAGE_RE.match(txt) and len(txt) <= 12 and not is_punct:
                continue
            if a.score < 0.62:
                continue
            if a.score < 0.72 and _NOISE_RE.match(txt) and not is_punct:
                continue

            # Check if line has a comparable-size neighbor in the same bubble
            has_neighbor = False
            has_close_vert_col = False
            for j, b in enumerate(lines):
                if i == j or not b.text.strip():
                    continue
                fs_ratio = max(a.font_size, b.font_size) / max(1.0, min(a.font_size, b.font_size))
                if fs_ratio > 2.0:
                    continue
                ref = min(max(a.font_size, b.font_size, 12.0), 42.0)
                if abs(a.cx - b.cx) < 2.2 * ref and abs(a.cy - b.cy) < 2.6 * ref:
                    has_neighbor = True
                    if (min(a.y1, b.y1) - max(a.y0, b.y0)) > 0.25 * min(a.h, b.h):
                        x_gap = max(0.0, max(a.x0 - b.x1, b.x0 - a.x1))
                        if x_gap < 0.9 * max(a.font_size, b.font_size):
                            has_close_vert_col = True

            # Drop single ASCII letter artifacts (keep 'I'/'A' on English pages, and pure punctuation near bubbles)
            if len(txt) == 1 and txt.isascii() and not is_punct:
                if (not mostly_ascii_page) or (not has_neighbor) or (a.font_size > 48.0) or (txt.upper() not in {"I", "A"}):
                    continue

            # Drop steeply tilted short artwork noise
            if len(txt) <= 2 and abs(a.angle_deg) > 22.0 and a.score < 0.90 and not is_punct:
                continue

            # Drop low-confidence standalone 1-char CJK noise on artwork (e.g. '川' on '///')
            if len(txt) == 1 and (not is_punct) and a.score < 0.82 and not has_close_vert_col:
                continue

            if not has_neighbor:
                if mostly_ascii_page and len(txt) <= 2 and not any(c.isascii() and c.isalpha() for c in txt) and not is_punct:
                    continue
                if txt.isascii() and len(txt) <= 5 and a.score < 0.78 and not is_punct:
                    continue
                if a.score < 0.65:
                    continue

            if img_bgr is not None and (
                (not has_neighbor and len(txt) <= 3 and a.score < 0.90)
                or (len(txt) <= 2 and a.score < 0.72 and not is_punct)
            ):
                ih, iw = img_bgr.shape[:2]
                rx0, ry0 = max(0, int(a.x0)), max(0, int(a.y0))
                rx1, ry1 = min(iw, int(a.x1)), min(ih, int(a.y1))
                if rx1 > rx0 and ry1 > ry0:
                    patch = cv2.cvtColor(img_bgr[ry0:ry1, rx0:rx1], cv2.COLOR_BGR2GRAY)
                    p85 = float(np.percentile(patch, 85))
                    # Real bubble interiors have dark ink strokes on a bright background
                    if p85 < 200.0 and float(np.percentile(patch, 15)) > 35.0:
                        continue
                    if (not has_neighbor) and len(txt) == 1 and (not is_punct) and p85 >= 200.0 and float(np.percentile(patch, 10)) > 185.0:
                        continue

            is_furi = False
            for j, b in enumerate(lines):
                if i == j:
                    continue
                if a.font_size >= 0.54 * b.font_size:
                    continue
                if a.is_vertical and b.is_vertical:
                    y_ov = min(a.y1, b.y1) - max(a.y0, b.y0)
                    x_gap = max(0.0, max(a.x0 - b.x1, b.x0 - a.x1))
                    if y_ov > 0.2 * a.h and x_gap < 1.3 * b.font_size:
                        if _KANA_ONLY_RE.match(txt) or len(txt) <= 8:
                            is_furi = True
                            break
                elif (not a.is_vertical) and (not b.is_vertical):
                    x_ov = min(a.x1, b.x1) - max(a.x0, b.x0)
                    y_gap = max(0.0, max(a.y0 - b.y1, b.y0 - a.y1))
                    if x_ov > 0.2 * a.w and y_gap < 1.3 * b.font_size:
                        if _KANA_ONLY_RE.match(txt) or len(txt) <= 8:
                            is_furi = True
                            break
            a.is_furigana = is_furi
            kept.append(a)
        return kept

    @staticmethod
    def _sample_line_bg_bgr(img_bgr: np.ndarray, ln: TextLine) -> np.ndarray | None:
        """Sample the surrounding bubble/box background color around a TextLine."""
        ih, iw = img_bgr.shape[:2]
        x0 = max(0, int(ln.x0) - 4)
        y0 = max(0, int(ln.y0) - 4)
        x1 = min(iw, int(ln.x1) + 4)
        y1 = min(ih, int(ln.y1) + 4)
        if x1 <= x0 or y1 <= y0:
            return None
        patch = img_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if float(np.median(gray)) >= 110.0:
            sel = patch[gray >= np.percentile(gray, 70)]
        else:
            sel = patch[gray <= np.percentile(gray, 30)]
        if sel.size == 0:
            return None
        return np.median(sel, axis=0).astype(np.float32)

    @classmethod
    def _bg_colors_compatible(
        cls, img_bgr: np.ndarray | None, a: TextLine, b: TextLine
    ) -> bool:
        if img_bgr is None:
            return True
        bg_a = cls._sample_line_bg_bgr(img_bgr, a)
        bg_b = cls._sample_line_bg_bgr(img_bgr, b)
        if bg_a is None or bg_b is None:
            return True
        return float(np.linalg.norm(bg_a - bg_b)) <= 48.0

    @classmethod
    def _has_horizontal_barrier_between(
        cls, img_bgr: np.ndarray | None, a: TextLine, b: TextLine
    ) -> bool:
        """Check for dark/colored outlines, borders, or artwork separating vertically stacked lines."""
        if img_bgr is None:
            return False
        top, bot = (a, b) if a.cy <= b.cy else (b, a)
        ih, iw = img_bgr.shape[:2]
        ov_x0 = max(int(top.x0), int(bot.x0))
        ov_x1 = min(int(top.x1), int(bot.x1))
        if ov_x1 > ov_x0 + 4:
            x0, x1 = max(0, ov_x0), min(iw, ov_x1)
        else:
            half = max(3, int(0.25 * min(top.w, bot.w)))
            x0 = max(0, int(min(top.cx, bot.cx)) - half)
            x1 = min(iw, int(max(top.cx, bot.cx)) + half)
        y0 = max(0, int(top.y1))
        y1 = min(ih, int(bot.y0))
        if x1 <= x0 + 2 or y1 <= y0 + 1:
            return False
        strip = img_bgr[y0:y1, x0:x1]
        gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        bg_top = cls._sample_line_bg_bgr(img_bgr, top)
        bg_bot = cls._sample_line_bg_bgr(img_bgr, bot)
        if bg_top is not None and bg_bot is not None:
            bg = 0.5 * (bg_top + bg_bot)
            if float(bg.mean()) > 140.0 and float(np.min(gray_strip.mean(axis=1))) < 165.0:
                return True
            dist = np.linalg.norm(strip.astype(np.float32) - bg, axis=2)
            row_bar_frac = (dist > 40.0).mean(axis=1)
            return bool(np.max(row_bar_frac) >= 0.35)
        return bool(np.min(gray_strip.mean(axis=1)) < 160.0)

    @classmethod
    def _has_vertical_barrier_between(
        cls, img_bgr: np.ndarray | None, a: TextLine, b: TextLine
    ) -> bool:
        """Check for bubble outlines, panel borders, or colored artwork separating adjacent columns."""
        if img_bgr is None:
            return False
        left, right = (a, b) if a.cx <= b.cx else (b, a)
        ih, iw = img_bgr.shape[:2]
        x0 = max(0, int(left.x1))
        x1 = min(iw, int(right.x0))
        y0 = max(0, int(max(left.y0, right.y0)))
        y1 = min(ih, int(min(left.y1, right.y1)))
        if y1 - y0 < 6:
            mid_y = int(0.5 * (y0 + y1))
            y0 = max(0, mid_y - 4)
            y1 = min(ih, mid_y + 4)
        if x1 <= x0 + 2 or y1 <= y0 + 4:
            return False
        strip = img_bgr[y0:y1, x0:x1]
        gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        bg_l = cls._sample_line_bg_bgr(img_bgr, left)
        bg_r = cls._sample_line_bg_bgr(img_bgr, right)
        if bg_l is not None and bg_r is not None:
            bg = 0.5 * (bg_l + bg_r)
            bg_lum = float(0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2])
            if bg_lum > 140.0 and float(np.min(gray_strip.mean(axis=0))) < 170.0:
                return True
            dist = np.linalg.norm(strip.astype(np.float32) - bg, axis=2)
            # Only darker outlines/borders/artwork count as walls (ignore brighter shine/highlights)
            dark_or_wall = (dist > 48.0) & (gray_strip.astype(np.float32) < bg_lum - 18.0)
            col_bar_frac = dark_or_wall.mean(axis=0)
            return bool(np.max(col_bar_frac) >= 0.35)
        return bool(np.min(gray_strip.mean(axis=0)) < 165.0)

    @classmethod
    def _should_merge(cls, a: TextLine, b: TextLine, img_bgr: np.ndarray | None = None) -> bool:
        """Orientation-aware, visual-barrier & bubble-color-aware merge check."""
        ref_fs = max(a.font_size, b.font_size)
        fs_ratio = max(a.font_size, b.font_size) / max(1.0, min(a.font_size, b.font_size))
        a_punct = bool(_PUNCT_ONLY_RE.match(a.text.strip()))
        b_punct = bool(_PUNCT_ONLY_RE.match(b.text.strip()))

        if a.is_vertical != b.is_vertical:
            # Attach trailing '?' / '!' directly beside a vertical CJK line in the same bubble
            if (a_punct and b.is_vertical) or (b_punct and a.is_vertical):
                y_ov = min(a.y1, b.y1) - max(a.y0, b.y0)
                x_gap = max(0.0, max(a.x0 - b.x1, b.x0 - a.x1))
                if (
                    fs_ratio <= 1.65
                    and y_ov > 0.20 * min(a.h, b.h)
                    and x_gap <= 0.55 * ref_fs
                    and cls._bg_colors_compatible(img_bgr, a, b)
                ):
                    if x_gap <= 2.0 or not cls._has_vertical_barrier_between(img_bgr, a, b):
                        return True
            return False

        ang_diff = abs(a.angle_deg - b.angle_deg) % 180.0
        ang_diff = min(ang_diff, 180.0 - ang_diff)
        if ang_diff > 20.0:
            return False
        short_line = len(a.text.strip()) <= 3 or len(b.text.strip()) <= 3
        if fs_ratio > (1.52 if short_line else 1.50):
            return False

        if a.is_vertical:
            y_ov = min(a.y1, b.y1) - max(a.y0, b.y0)
            x_gap = max(0.0, max(a.x0 - b.x1, b.x0 - a.x1))
            y_gap = max(0.0, max(a.y0 - b.y1, b.y0 - a.y1))
            same_col = abs(a.cx - b.cx) < 0.55 * ref_fs and y_gap < 1.15 * ref_fs
            adj_col = y_ov > 0.18 * min(a.h, b.h) and x_gap <= 1.05 * ref_fs
            if not (same_col or adj_col):
                return False
            if not cls._bg_colors_compatible(img_bgr, a, b):
                return False
            if adj_col and x_gap > 2.0 and cls._has_vertical_barrier_between(img_bgr, a, b):
                return False
            if same_col and (not adj_col) and y_gap > 2.0 and cls._has_horizontal_barrier_between(img_bgr, a, b):
                return False
            return True
        else:
            x_ov = min(a.x1, b.x1) - max(a.x0, b.x0)
            y_gap = max(0.0, max(a.y0 - b.y1, b.y0 - a.y1))
            center_aligned = abs(a.cx - b.cx) <= 0.55 * max(a.w, b.w)
            horiz_ok = (x_ov > 0.12 * min(a.w, b.w)) or center_aligned
            if not horiz_ok or y_gap > 1.35 * ref_fs:
                return False
            if not cls._bg_colors_compatible(img_bgr, a, b):
                return False
            if y_gap > max(3.0, 0.30 * ref_fs) and cls._has_horizontal_barrier_between(img_bgr, a, b):
                return False
            return True

    @staticmethod
    def _sort_horizontal_group(grp: List[TextLine]) -> List[TextLine]:
        """Sort horizontal lines strictly top-to-bottom by row, left-to-right within each row."""
        rows: List[List[TextLine]] = []
        for ln in sorted(grp, key=lambda l: l.cy):
            placed = False
            for row in rows:
                row_cy = float(np.mean([r.cy for r in row]))
                ref_h = min(ln.font_size, float(np.mean([r.font_size for r in row])))
                if abs(ln.cy - row_cy) <= 0.45 * ref_h:
                    row.append(ln)
                    placed = True
                    break
            if not placed:
                rows.append([ln])
        rows.sort(key=lambda r: float(np.mean([l.cy for l in r])))
        ordered: List[TextLine] = []
        for row in rows:
            row.sort(key=lambda l: l.cx)
            ordered.extend(row)
        return ordered

    @staticmethod
    def _sort_vertical_group(grp: List[TextLine]) -> List[TextLine]:
        """Sort vertical manga lines strictly right-to-left by column, top-to-bottom within column."""
        cols: List[List[TextLine]] = []
        for ln in sorted(grp, key=lambda l: -l.cx):
            placed = False
            for col in cols:
                col_cx = float(np.mean([c.cx for c in col]))
                ref_w = min(ln.font_size, float(np.mean([c.font_size for c in col])))
                y_overlap_conflict = any(
                    (min(ln.y1, c.y1) - max(ln.y0, c.y0)) >= 0.35 * min(ln.h, c.h)
                    for c in col
                )
                if (not y_overlap_conflict) and abs(ln.cx - col_cx) <= 0.42 * ref_w:
                    col.append(ln)
                    placed = True
                    break
            if not placed:
                cols.append([ln])
        cols.sort(key=lambda c: -float(np.mean([l.cx for l in c])))
        ordered: List[TextLine] = []
        for col in cols:
            col.sort(key=lambda l: l.cy)
            ordered.extend(col)
        return ordered

    @staticmethod
    def _prep_padded(det_td, img: np.ndarray, target_side: int) -> Tuple[np.ndarray, int, int] | None:
        if img is None or img.size == 0:
            return None
        h0, w0 = img.shape[:2]
        if max(h0, w0) > target_side:
            s = target_side / max(h0, w0)
            img = cv2.resize(
                img,
                (max(32, int(round(w0 * s))), max(32, int(round(h0 * s)))),
                interpolation=cv2.INTER_AREA,
            )
        try:
            op = det_td.get_preprocess()
        except TypeError:
            op = det_td.get_preprocess(max(img.shape[0], img.shape[1]))
        arr = op(img)
        if arr is None:
            return None
        _, _, h, w = arr.shape
        if h > target_side or w > target_side:
            arr = arr[:, :, :target_side, :target_side]
            h, w = arr.shape[2], arr.shape[3]
        if h == target_side and w == target_side:
            return arr, h, w
        pad_h = max(0, target_side - h)
        pad_w = max(0, target_side - w)
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
        return arr, h, w

    def extract(
        self,
        image_bgr: np.ndarray,
        engine: RapidOCR | None = None,
        ocr_model: str = "auto",
    ) -> Tuple[List[TextBlock], List[np.ndarray]]:
        """
        Run OCR + orientation recovery + furigana filtering + SFX-aware clustering.
        ocr_model: 'auto' | 'medium' (high accuracy) | 'tiny' (fastest).
        Returns (text_blocks, all_inpaint_polygons).
        """
        if engine is not None:
            det_eng = engine
            rec_eng = engine
        elif ocr_model == "tiny":
            det_eng = self.fast_det_engine
            rec_eng = self.fast_det_engine
        else:
            det_eng = self.det_engine
            rec_eng = self.rec_engine
        det_td = self._ensure_det_td(det_eng)
        rec_tr = self._ensure_rec_tr(rec_eng)
        target_side = 800 if (self._fast_det_engine is not None and det_eng is self._fast_det_engine) else OCR_DET_LIMIT_SIDE
        prep = self._prep_padded(det_td, image_bgr, target_side)
        if prep is None:
            return [], []
        padded, ph, pw = prep
        pred = det_td.session(padded)[:, :, :ph, :pw]
        boxes, _ = det_td.postprocess_op(pred, (image_bgr.shape[0], image_bgr.shape[1]))
        if boxes is None or len(boxes) == 0:
            return [], []
        boxes = det_td.sorted_boxes(boxes)

        crops = [self._crop_poly(image_bgr, np.asarray(b, dtype=np.float32)) for b in boxes]
        rec_res = rec_tr(TextRecInput(img=crops))
        if not rec_res or not rec_res.txts:
            return [], []

        return self._build_blocks(
            image_bgr, boxes, rec_res.txts, rec_res.scores, rec_engine=rec_eng
        )

    def _build_blocks(
        self,
        image_bgr: np.ndarray,
        boxes,
        txts,
        scores,
        rec_engine: RapidOCR | None = None,
    ) -> Tuple[List[TextBlock], List[np.ndarray]]:
        raw_lines: List[TextLine] = []
        for box, txt, score in zip(boxes, txts, scores):
            poly = np.asarray(box, dtype=np.float32)
            x0, y0 = float(poly[:, 0].min()), float(poly[:, 1].min())
            x1, y1 = float(poly[:, 0].max()), float(poly[:, 1].max())
            w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
            clean_txt = (txt or "").strip()
            is_vert = (h > 1.35 * w) and not (clean_txt.isascii() and (len(clean_txt) <= 3 or h < 2.8 * w))
            fs = w if is_vert else h
            raw_lines.append(
                TextLine(
                    poly=poly,
                    text=clean_txt,
                    score=float(score),
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    cx=0.5 * (x0 + x1),
                    cy=0.5 * (y0 + y1),
                    w=w,
                    h=h,
                    is_vertical=is_vert,
                    font_size=fs,
                    angle_deg=self._poly_angle(poly),
                )
            )

        self._retry_rotated_crops(image_bgr, raw_lines, engine=rec_engine)
        lines = self._mark_furigana_and_noise(raw_lines, image_bgr)
        kept_ids = {id(ln) for ln in lines}
        all_inpaint_polys = [ln.poly for ln in lines]
        # Erase stylized/low-confidence companion columns inside kept vertical bubbles
        for raw_ln in raw_lines:
            if id(raw_ln) in kept_ids or raw_ln.score < 0.15 or len(raw_ln.text.strip()) < 2:
                continue
            for kept_ln in lines:
                if kept_ln.score < 0.85 or not kept_ln.is_vertical:
                    continue
                y_ov = min(raw_ln.y1, kept_ln.y1) - max(raw_ln.y0, kept_ln.y0)
                x_gap = max(0.0, max(raw_ln.x0 - kept_ln.x1, kept_ln.x0 - raw_ln.x1))
                ref_fs = max(raw_ln.font_size, kept_ln.font_size)
                if (
                    y_ov >= 0.45 * min(raw_ln.h, kept_ln.h)
                    and x_gap <= 0.35 * ref_fs
                    and self._bg_colors_compatible(image_bgr, raw_ln, kept_ln)
                ):
                    all_inpaint_polys.append(raw_ln.poly)
                    break

        main_lines = [ln for ln in lines if not ln.is_furigana]
        if not main_lines:
            return [], all_inpaint_polys

        # Union-Find clustering
        n = len(main_lines)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(main_lines[i], main_lines[j], image_bgr):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri

        groups: dict[int, List[TextLine]] = {}
        for i, ln in enumerate(main_lines):
            groups.setdefault(find(i), []).append(ln)

        blocks: List[TextBlock] = []
        for idx, grp in enumerate(groups.values()):
            content_lines = [l for l in grp if not _PUNCT_ONLY_RE.match(l.text.strip())] or grp
            is_vert = sum(1 for l in content_lines if l.is_vertical) > len(content_lines) / 2
            if is_vert:
                grp = self._sort_vertical_group(grp)
            else:
                grp = self._sort_horizontal_group(grp)

            joined = " ".join(l.text for l in grp) if not is_vert else "".join(l.text for l in grp)
            if not is_vert:
                joined = re.sub(r"(?<![\u4e00-\u9fff])工(?=\s+[A-Za-z])", "I", joined)
                joined = re.sub(r"([A-Za-z])-\s+([A-Za-z'])", r"\1\2", joined)
            bx0 = int(min(l.x0 for l in grp))
            by0 = int(min(l.y0 for l in grp))
            bx1 = int(max(l.x1 for l in grp))
            by1 = int(max(l.y1 for l in grp))
            avg_ang = float(np.mean([l.angle_deg for l in grp]))
            avg_fs = float(np.mean([l.font_size for l in grp]))
            avg_conf = float(np.mean([l.score for l in grp]))
            is_sfx = len(grp) == 1 and (abs(avg_ang) > 8.0 or avg_fs > 36.0)

            blocks.append(
                TextBlock(
                    id=idx,
                    lines=grp,
                    text=joined,
                    bbox=(bx0, by0, bx1, by1),
                    polygon=np.array(
                        [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], dtype=np.float32
                    ),
                    inpaint_polygons=[l.poly for l in grp],
                    is_vertical=is_vert,
                    is_sfx=is_sfx,
                    angle_deg=avg_ang,
                    avg_font_size=avg_fs,
                    confidence=avg_conf,
                )
            )

        # Sort blocks in natural manga reading order (top-to-bottom, right-to-left)
        blocks.sort(key=lambda b: (b.bbox[1] // 150, -b.bbox[0]))
        for i, b in enumerate(blocks):
            b.id = i

        return blocks, all_inpaint_polys

    def extract_batch(
        self,
        images_bgr: List[np.ndarray],
        workers: int = 6,
        ocr_model: str = "auto",
    ) -> List[Tuple[List[TextBlock], List[np.ndarray]]]:
        """Ultra-fast batched GPU Det + parallel CPU Rec across a batch/chapter of images."""
        n = len(images_bgr)
        if n == 0:
            return []
        if n == 1:
            return [self.extract(images_bgr[0], ocr_model=ocr_model)]

        use_tiny = (ocr_model == "tiny")
        det_eng = self.fast_det_engine if use_tiny else self.det_engine
        target_side = 800 if use_tiny else OCR_DET_LIMIT_SIDE
        det_td = self._ensure_det_td(det_eng)
        rec_pool = self.ensure_fast_pool(workers=min(4, max(2, workers)), fast_mode=use_tiny)

        # 1 & 2. Windowed CPU preprocess + GPU Det (frees 18.8 MiB float32 tensors immediately)
        det_preds: List[np.ndarray | None] = [None] * n
        step = 4 if use_tiny else 1
        prep_window = max(step, 4)
        for w_start in range(0, n, prep_window):
            w_end = min(n, w_start + prep_window)
            with ThreadPoolExecutor(max_workers=min(4, w_end - w_start)) as ex:
                pre_win = list(
                    ex.map(lambda im: self._prep_padded(det_td, im, target_side), images_bgr[w_start:w_end])
                )
            valid_local = [k for k, p in enumerate(pre_win) if p is not None]
            for s in range(0, len(valid_local), step):
                chunk_k = valid_local[s : s + step]
                batch_np = (
                    pre_win[chunk_k[0]][0]
                    if len(chunk_k) == 1
                    else np.concatenate([pre_win[k][0] for k in chunk_k], axis=0)
                )
                out = det_td.session(batch_np)
                del batch_np
                for pos, k in enumerate(chunk_k):
                    _, ph, pw = pre_win[k]
                    det_preds[w_start + k] = out[pos : pos + 1, :, :ph, :pw].copy()
                    pre_win[k] = None
                del out
            del pre_win

        # 3. Parallel CPU DBPostProcess + perspective crop
        def _post_and_crop(idx: int):
            pred = det_preds[idx]
            det_preds[idx] = None
            if pred is None:
                return [], []
            img = images_bgr[idx]
            boxes, _ = det_td.postprocess_op(pred, (img.shape[0], img.shape[1]))
            del pred
            if boxes is None or len(boxes) == 0:
                return [], []
            boxes = det_td.sorted_boxes(boxes)
            crops = [self._crop_poly(img, np.asarray(b, dtype=np.float32)) for b in boxes]
            return boxes, crops

        with ThreadPoolExecutor(max_workers=8) as ex:
            post_results = list(ex.map(_post_and_crop, range(n)))
        del det_preds

        # Flatten all crops for parallel Rec
        flat_crops: List[np.ndarray] = []
        page_ranges: List[Tuple[int, int]] = []
        for idx, (boxes, crops) in enumerate(post_results):
            s = len(flat_crops)
            flat_crops.extend(crops)
            page_ranges.append((s, len(flat_crops)))
            post_results[idx] = (boxes, [])

        flat_txts: List[str] = [""] * len(flat_crops)
        flat_scores: List[float] = [0.0] * len(flat_crops)
        if flat_crops:
            # Sort crops by aspect ratio so Rec batches minimize wasted width padding
            order = sorted(
                range(len(flat_crops)),
                key=lambda k: flat_crops[k].shape[1] / max(1, flat_crops[k].shape[0]),
            )
            num_w = len(rec_pool)
            chunk_sz = max(8, math.ceil(len(order) / num_w))
            rec_chunks = [
                (
                    w_idx,
                    order[s : s + chunk_sz],
                    [flat_crops[idx_k] for idx_k in order[s : s + chunk_sz]],
                )
                for w_idx, s in enumerate(range(0, len(order), chunk_sz))
            ]

            def _run_rec(item: Tuple[int, List[int], List[np.ndarray]]):
                w_idx, orig_idxs, crops_sub = item
                rec_tr = self._ensure_rec_tr(rec_pool[w_idx % num_w])
                res = rec_tr(TextRecInput(img=crops_sub))
                return orig_idxs, res

            with ThreadPoolExecutor(max_workers=num_w) as ex:
                for orig_idxs, res in ex.map(_run_rec, rec_chunks):
                    if res and res.txts:
                        for orig_k, t, sc in zip(orig_idxs, res.txts, res.scores):
                            flat_txts[orig_k] = t
                            flat_scores[orig_k] = float(sc)
            flat_crops.clear()

        # 4. Parallel layout clustering & 180-deg flip recovery per page
        def _finalize(idx: int) -> Tuple[List[TextBlock], List[np.ndarray]]:
            boxes, _ = post_results[idx]
            s, e = page_ranges[idx]
            if s == e:
                return [], []
            return self._build_blocks(
                images_bgr[idx],
                boxes,
                flat_txts[s:e],
                flat_scores[s:e],
                rec_engine=rec_pool[idx % len(rec_pool)],
            )

        with ThreadPoolExecutor(max_workers=max(1, len(rec_pool))) as ex:
            return list(ex.map(_finalize, range(n)))
