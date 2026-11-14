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
_NUM_GARBAGE_RE = re.compile(r"^[\d\s.,:%/()\[\]-]+$")


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

    def __init__(self, use_cuda: bool | None = None) -> None:
        if use_cuda is None:
            try:
                import onnxruntime as ort

                use_cuda = torch.cuda.is_available() and (
                    "CUDAExecutionProvider" in ort.get_available_providers()
                )
            except Exception:
                use_cuda = False
        self.use_cuda = bool(use_cuda)
        self._lock = threading.Lock()
        try:
            self.det_engine = self._build_det_engine(use_cuda=self.use_cuda, fast_mode=False)
        except Exception:
            if not self.use_cuda:
                raise
            self.use_cuda = False
            self.det_engine = self._build_det_engine(use_cuda=False, fast_mode=False)
        self._fast_det_engine: RapidOCR | None = None
        self.rec_engine = self._build_rec_engine(fast_mode=False)
        self._rec_pool: List[RapidOCR] = []

    @property
    def fast_det_engine(self) -> RapidOCR:
        if self._fast_det_engine is None:
            with self._lock:
                if self._fast_det_engine is None:
                    self._fast_det_engine = self._build_det_engine(
                        use_cuda=self.use_cuda, fast_mode=True
                    )
        return self._fast_det_engine

    @staticmethod
    def _build_det_engine(use_cuda: bool, fast_mode: bool = False) -> RapidOCR:
        return RapidOCR(
            params={
                "EngineConfig.onnxruntime.use_cuda": use_cuda,
                "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "HEURISTIC",
                "Global.use_cls": False,
                "Det.model_type": ModelType.TINY if fast_mode else ModelType.MEDIUM,
                "Rec.model_type": ModelType.TINY if fast_mode else ModelType.SMALL,
                "Det.limit_type": "max",
                "Det.limit_side_len": 800 if fast_mode else 1280,
                "Det.box_thresh": OCR_DET_BOX_THRESH,
                "Det.thresh": OCR_DET_THRESH,
                "Det.unclip_ratio": OCR_DET_UNCLIP_RATIO,
            }
        )

    @staticmethod
    def _build_rec_engine(fast_mode: bool = False) -> RapidOCR:
        return RapidOCR(
            params={
                "EngineConfig.onnxruntime.use_cuda": False,
                "Global.use_cls": False,
                "Det.model_type": ModelType.TINY,
                "Rec.model_type": ModelType.TINY if fast_mode else ModelType.SMALL,
            }
        )

    def ensure_fast_pool(self, workers: int = 4) -> List[RapidOCR]:
        """Lazily initialize a thread-safe pool of CPU Rec workers for batch OCR."""
        while len(self._rec_pool) < workers:
            self._rec_pool.append(self._build_rec_engine(fast_mode=True))
        return self._rec_pool[:workers]

    @staticmethod
    def _poly_angle(poly: np.ndarray) -> float:
        """Compute dominant orientation angle in degrees (-90 to 90)."""
        p0, p1, p2, p3 = poly
        e_top = p1 - p0
        e_side = p3 - p0
        if np.linalg.norm(e_top) >= np.linalg.norm(e_side):
            ang = math.degrees(math.atan2(e_top[1], e_top[0]))
        else:
            ang = math.degrees(math.atan2(e_side[1], e_side[0]))
        while ang > 90:
            ang -= 180
        while ang < -90:
            ang += 180
        return ang

    @staticmethod
    def _crop_poly(img_bgr: np.ndarray, poly: np.ndarray) -> np.ndarray:
        pts = poly.astype(np.float32)
        w = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
        h = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
        w, h = max(w, 4), max(h, 4)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        crop = cv2.warpPerspective(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if h >= w * 1.5:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return crop

    def _retry_rotated_crops(
        self, img_bgr: np.ndarray, lines: List[TextLine], engine: RapidOCR | None = None
    ) -> None:
        """For low-confidence vertical/sideways crops (e.g. rotated English), test 180-deg flip."""
        candidates: List[Tuple[int, np.ndarray]] = []
        for idx, ln in enumerate(lines):
            # Sideways English or upside-down text often scores < 0.86 when rotated wrong
            if ln.score < 0.88 and (ln.is_vertical or abs(ln.angle_deg) > 15):
                crop = self._crop_poly(img_bgr, ln.poly)
                flipped = cv2.rotate(crop, cv2.ROTATE_180)
                candidates.append((idx, flipped))

        if not candidates:
            return

        crops = [c[1] for c in candidates]
        rec_engine = engine or self.rec_engine
        rec_res = rec_engine.text_rec(TextRecInput(img=crops))
        if not rec_res or not rec_res.txts:
            return

        for (idx, _), new_txt, new_score in zip(candidates, rec_res.txts, rec_res.scores):
            new_txt = (new_txt or "").strip()
            if new_txt and new_score > lines[idx].score + 0.05:
                lines[idx].text = new_txt
                lines[idx].score = float(new_score)

    @staticmethod
    def _mark_furigana_and_noise(lines: List[TextLine]) -> List[TextLine]:
        """Flag tiny furigana ruby text beside main lines and drop pure graphic noise."""
        kept: List[TextLine] = []
        for i, a in enumerate(lines):
            txt = a.text.strip()
            if not txt:
                continue
            # Drop screentone/artwork numeric & low-confidence ASCII false positives
            if _NUM_GARBAGE_RE.match(txt) and len(txt) <= 12:
                continue
            if a.score < 0.45 and len(txt) <= 2:
                continue
            if a.score < 0.72 and _NOISE_RE.match(txt):
                continue
            if txt.isascii() and len(txt) <= 5 and a.score < 0.85:
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
    def _should_merge(a: TextLine, b: TextLine) -> bool:
        """Orientation-aware & scale-aware merge check so clustered SFX stay separate."""
        if a.is_vertical != b.is_vertical:
            return False
        ang_diff = abs(a.angle_deg - b.angle_deg) % 180.0
        ang_diff = min(ang_diff, 180.0 - ang_diff)
        if ang_diff > 18.0:
            return False
        fs_ratio = max(a.font_size, b.font_size) / max(1.0, min(a.font_size, b.font_size))
        if fs_ratio > 1.35:
            return False

        ref_fs = min(a.font_size, b.font_size)
        if a.is_vertical:
            y_ov = min(a.y1, b.y1) - max(a.y0, b.y0)
            x_gap = max(0.0, max(a.x0 - b.x1, b.x0 - a.x1))
            # Same vertical column split into top/bottom segments OR adjacent columns in same bubble
            same_col = abs(a.cx - b.cx) < 0.65 * ref_fs and max(0.0, max(a.y0 - b.y1, b.y0 - a.y1)) < 1.8 * ref_fs
            adj_col = y_ov > 0.15 * min(a.h, b.h) and x_gap < 1.75 * ref_fs
            return same_col or adj_col
        else:
            x_ov = min(a.x1, b.x1) - max(a.x0, b.x0)
            y_gap = max(0.0, max(a.y0 - b.y1, b.y0 - a.y1))
            return x_ov > 0.25 * min(a.w, b.w) and y_gap < 0.55 * ref_fs

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
        # Group lines into columns if their horizontal centers are within 0.6 * min(font_size)
        cols: List[List[TextLine]] = []
        for ln in sorted(grp, key=lambda l: -l.cx):
            placed = False
            for col in cols:
                col_cx = float(np.mean([c.cx for c in col]))
                ref_w = min(ln.font_size, float(np.mean([c.font_size for c in col])))
                if abs(ln.cx - col_cx) <= 0.65 * ref_w:
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
        op = det_td.get_preprocess(max(img.shape[0], img.shape[1]))
        arr = op(img)
        if arr is None:
            return None
        _, _, h, w = arr.shape
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
        det_td = det_eng.text_det
        target_side = 800 if (self._fast_det_engine is not None and det_eng is self._fast_det_engine) else 1280
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
        rec_res = rec_eng.text_rec(TextRecInput(img=crops))
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
            is_vert = h > 1.35 * w
            fs = w if is_vert else h
            raw_lines.append(
                TextLine(
                    poly=poly,
                    text=(txt or "").strip(),
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
        lines = self._mark_furigana_and_noise(raw_lines)
        all_inpaint_polys = [ln.poly for ln in lines]

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
                if self._should_merge(main_lines[i], main_lines[j]):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri

        groups: dict[int, List[TextLine]] = {}
        for i, ln in enumerate(main_lines):
            groups.setdefault(find(i), []).append(ln)

        blocks: List[TextBlock] = []
        for idx, grp in enumerate(groups.values()):
            is_vert = sum(1 for l in grp if l.is_vertical) > len(grp) / 2
            if is_vert:
                grp = self._sort_vertical_group(grp)
            else:
                grp = self._sort_horizontal_group(grp)

            joined = " ".join(l.text for l in grp) if not is_vert else "".join(l.text for l in grp)
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

        use_tiny = ocr_model in {"auto", "tiny"}
        det_eng = self.fast_det_engine if use_tiny else self.det_engine
        target_side = 800 if use_tiny else 1280
        det_td = det_eng.text_det
        rec_pool = self.ensure_fast_pool(workers=min(4, max(2, workers)))

        # 1. Parallel CPU preprocess + pad to fixed square tensor so CUDA never re-tunes shapes
        with ThreadPoolExecutor(max_workers=8) as ex:
            pre_list = list(ex.map(lambda im: self._prep_padded(det_td, im, target_side), images_bgr))

        # 2. Run batched GPU Det (batch<=16) on unified (B, 3, 800, 800) tensor
        valid_idxs = [i for i, p in enumerate(pre_list) if p is not None]
        det_preds: List[np.ndarray | None] = [None] * n
        for start in range(0, len(valid_idxs), 16):
            chunk_idxs = valid_idxs[start : start + 16]
            batch_np = np.concatenate([pre_list[i][0] for i in chunk_idxs], axis=0)
            out = det_td.session(batch_np)
            for k, orig_i in enumerate(chunk_idxs):
                _, ph, pw = pre_list[orig_i]
                det_preds[orig_i] = out[k : k + 1, :, :ph, :pw]

        # 3. Parallel CPU DBPostProcess + perspective crop
        def _post_and_crop(idx: int):
            pred = det_preds[idx]
            if pred is None:
                return [], []
            img = images_bgr[idx]
            boxes, _ = det_td.postprocess_op(pred, (img.shape[0], img.shape[1]))
            if boxes is None or len(boxes) == 0:
                return [], []
            boxes = det_td.sorted_boxes(boxes)
            crops = [self._crop_poly(img, np.asarray(b, dtype=np.float32)) for b in boxes]
            return boxes, crops

        with ThreadPoolExecutor(max_workers=8) as ex:
            post_results = list(ex.map(_post_and_crop, range(n)))

        # Flatten all crops for parallel Rec
        flat_crops: List[np.ndarray] = []
        page_ranges: List[Tuple[int, int]] = []
        for boxes, crops in post_results:
            s = len(flat_crops)
            flat_crops.extend(crops)
            page_ranges.append((s, len(flat_crops)))

        flat_txts: List[str] = [""] * len(flat_crops)
        flat_scores: List[float] = [0.0] * len(flat_crops)
        if flat_crops:
            chunk_sz = max(8, math.ceil(len(flat_crops) / len(rec_pool)))
            rec_chunks = [
                (w_idx, s, flat_crops[s : s + chunk_sz])
                for w_idx, s in enumerate(range(0, len(flat_crops), chunk_sz))
            ]

            def _run_rec(item: Tuple[int, int, List[np.ndarray]]):
                w_idx, offset, crops_sub = item
                res = rec_pool[w_idx % len(rec_pool)].text_rec(TextRecInput(img=crops_sub))
                return offset, res

            with ThreadPoolExecutor(max_workers=len(rec_pool)) as ex:
                for offset, res in ex.map(_run_rec, rec_chunks):
                    if res and res.txts:
                        for j, (t, sc) in enumerate(zip(res.txts, res.scores)):
                            flat_txts[offset + j] = t
                            flat_scores[offset + j] = float(sc)

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

        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(_finalize, range(n)))
