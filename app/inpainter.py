"""Smart hybrid text removal: instant solid-fill for uniform bubbles + mayocream/aot-inpainting on GPU."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from app.config import (
    AOT_MAX_SIDE,
    AOT_REPO_ID,
    UNIFORM_INLIER_RATIO,
    UNIFORM_STD_THRESH,
)


def _relu_nf(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) * 1.7139588594436646


def _my_layer_norm(feat: torch.Tensor) -> torch.Tensor:
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    return 5.0 * (2.0 * (feat - mean) / std - 1.0)


class _LambdaLayer(nn.Module):
    def __init__(self, f) -> None:
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class _ScaledWSConv2d(nn.Conv2d):
    """2D Conv layer with Scaled Weight Standardization matching mayocream/aot-inpainting."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        eps: float = 1e-4,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.eps = eps

    def get_weight(self) -> torch.Tensor:
        fan_in = float(np.prod(self.weight.shape[1:]))
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdim=True)
        eps_t = torch.tensor(self.eps, device=var.device, dtype=var.dtype)
        scale = torch.rsqrt(torch.maximum(var * fan_in, eps_t)) * self.gain.view_as(var)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class _ScaledWSTransposeConv2d(nn.ConvTranspose2d):
    """2D Transpose Conv layer with Scaled Weight Standardization matching mayocream/aot-inpainting."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.gain = nn.Parameter(torch.ones(self.in_channels, 1, 1, 1))
        self.eps = eps

    def get_weight(self) -> torch.Tensor:
        fan_in = float(np.prod(self.weight.shape[1:]))
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdim=True)
        eps_t = torch.tensor(self.eps, device=var.device, dtype=var.dtype)
        scale = torch.rsqrt(torch.maximum(var * fan_in, eps_t)) * self.gain.view_as(var)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation
        )
        return F.conv_transpose2d(
            x,
            self.get_weight(),
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class _GatedWSConvPadded(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.padding = nn.ReflectionPad2d(((ks - 1) * dilation) // 2)
        self.conv = _ScaledWSConv2d(in_ch, out_ch, kernel_size=ks, stride=stride, dilation=dilation)
        self.conv_gate = _ScaledWSConv2d(
            in_ch, out_ch, kernel_size=ks, stride=stride, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding(x)
        signal = self.conv(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return signal * gate * 1.8


class _GatedWSTransposeConvPadded(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int, stride: int = 1) -> None:
        super().__init__()
        pad = (ks - 1) // 2
        self.conv = _ScaledWSTransposeConv2d(
            in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad
        )
        self.conv_gate = _ScaledWSTransposeConv2d(
            in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = self.conv(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return signal * gate * 1.8


class _AOTBlock(nn.Module):
    def __init__(self, dim: int, rates: List[int]) -> None:
        super().__init__()
        self.rates = rates
        branch_dim = dim // len(rates)
        for i, rate in enumerate(rates):
            setattr(
                self,
                f"block{i:02d}",
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, branch_dim, 3, padding=0, dilation=rate),
                    nn.ReLU(inplace=True),
                ),
            )
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1),
        )
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [getattr(self, f"block{i:02d}")(x) for i in range(len(self.rates))]
        out = self.fuse(torch.cat(branches, dim=1))
        mask = torch.sigmoid(_my_layer_norm(self.gate(x)))
        return x * (1.0 - mask) + out * mask


class AOTGenerator(nn.Module):
    """Exact AOT-GAN generator matching mayocream/aot-inpainting weights."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
        num_blocks: int = 10,
        dilation_rates: List[int] | None = None,
    ) -> None:
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [2, 4, 8, 16]
        c = base_channels
        self.head = nn.Sequential(
            _GatedWSConvPadded(in_channels, c, 3, stride=1),
            _LambdaLayer(_relu_nf),
            _GatedWSConvPadded(c, c * 2, 4, stride=2),
            _LambdaLayer(_relu_nf),
            _GatedWSConvPadded(c * 2, c * 4, 4, stride=2),
        )
        self.body_conv = nn.Sequential(
            *[_AOTBlock(c * 4, dilation_rates) for _ in range(num_blocks)]
        )
        self.tail = nn.Sequential(
            _GatedWSConvPadded(c * 4, c * 4, 3, 1),
            _LambdaLayer(_relu_nf),
            _GatedWSConvPadded(c * 4, c * 4, 3, 1),
            _LambdaLayer(_relu_nf),
            _GatedWSTransposeConvPadded(c * 4, c * 2, 4, 2),
            _LambdaLayer(_relu_nf),
            _GatedWSTransposeConvPadded(c * 2, c, 4, 2),
            _LambdaLayer(_relu_nf),
            _GatedWSConvPadded(c, out_channels, 3, stride=1),
        )

    def forward(self, mask_and_img: torch.Tensor) -> torch.Tensor:
        x = self.head(mask_and_img)
        x = self.body_conv(x)
        x = self.tail(x)
        return torch.clamp(x, -1.0, 1.0)


class SmartInpainter:
    """Hybrid inpainter: fast solid-color fill for uniform bubbles + AOT-GAN for complex artwork."""

    def __init__(self, device: str | None = None, lazy: bool = True) -> None:
        self.device_label = "cpu"
        self._use_ort_dml = False
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.device_label = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                self.device_label = "mps"
            else:
                self.device = "cpu"
                try:
                    import torch_directml  # type: ignore

                    if torch_directml.is_available():
                        self.device = torch_directml.device()
                        self.device_label = "dml"
                except Exception:
                    pass
                if self.device_label == "cpu":
                    try:
                        import onnxruntime as ort

                        if "DmlExecutionProvider" in ort.get_available_providers():
                            self.device_label = "dml"
                            self._use_ort_dml = True
                    except Exception:
                        pass
        else:
            self.device = device
            self.device_label = str(device)
        self.use_fp16 = self.device_label == "cuda"
        self._model_lock = threading.Lock()
        self._model: Optional[AOTGenerator] = None
        self._ort_session: Any = None
        self._ort_input_name: str = "input"
        self.pad_multiple: int = 8
        if not lazy:
            self.ensure_loaded()

    def ensure_loaded(self) -> Optional[AOTGenerator]:
        if self._model is not None or self._ort_session is not None:
            return self._model
        with self._model_lock:
            if self._model is None and self._ort_session is None:
                model, pad_mult, weights_path = self._load_aot_model()
                self.pad_multiple = pad_mult
                if self._use_ort_dml:
                    try:
                        self._ort_session = self._build_ort_dml_session(model, weights_path)
                        self._ort_input_name = self._ort_session.get_inputs()[0].name
                        # Free PyTorch weights when running via DirectML ONNX Runtime
                        return None
                    except Exception as exc:
                        print(f"[SmartInpainter] DirectML init failed, falling back to CPU: {exc}")
                        self._use_ort_dml = False
                        self.device_label = "cpu"
                self._model = model
        return self._model

    @property
    def model(self) -> AOTGenerator:
        res = self.ensure_loaded()
        if res is None:
            model, pad_mult, _ = self._load_aot_model()
            self.pad_multiple = pad_mult
            return model
        return res

    @staticmethod
    def _build_ort_dml_session(model: AOTGenerator, weights_path: str):
        import onnxruntime as ort

        onnx_path = Path(weights_path).with_name("aot_inpainting_opset17.onnx")
        if not onnx_path.exists() or onnx_path.stat().st_size < 100_000:
            tmp_path = onnx_path.with_suffix(".onnx.tmp")
            dummy = torch.zeros(1, 4, 128, 128, dtype=torch.float32)
            torch.onnx.export(
                model.cpu().eval(),
                dummy,
                str(tmp_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
                opset_version=17,
                dynamo=False,
            )
            tmp_path.replace(onnx_path)
        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(
            str(onnx_path),
            sess_options=so,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )

    def _load_aot_model(self) -> Tuple[AOTGenerator, int, str]:
        cfg_path = hf_hub_download(AOT_REPO_ID, "config.json")
        weights_path = hf_hub_download(AOT_REPO_ID, "model.safetensors")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = AOTGenerator(
            in_channels=cfg.get("input_channels", 4),
            out_channels=cfg.get("output_channels", 3),
            base_channels=cfg.get("base_channels", 32),
            num_blocks=cfg.get("num_blocks", 10),
            dilation_rates=cfg.get("dilation_rates", [2, 4, 8, 16]),
        )
        state = load_file(weights_path)
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        if self.use_fp16:
            model.half()
        return model, int(cfg.get("pad_multiple", 8)), weights_path

    @staticmethod
    def _extract_region_mask(
        crop_bgr: np.ndarray, poly_local: np.ndarray
    ) -> np.ndarray:
        """Build a tight pixel mask of text strokes (dark or light) inside a polygon."""
        h, w = crop_bgr.shape[:2]
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly_local.astype(np.int32)], 255)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        ring = cv2.dilate(poly_mask, np.ones((9, 9), np.uint8)) - poly_mask
        ring_vals = gray[ring > 0]
        bg_med = float(np.median(ring_vals)) if ring_vals.size > 0 else float(np.median(gray))

        if bg_med >= 128:
            _, stroke = cv2.threshold(
                gray, min(200, int(bg_med * 0.78)), 255, cv2.THRESH_BINARY_INV
            )
        else:
            _, stroke = cv2.threshold(
                gray, max(55, int(bg_med + (255 - bg_med) * 0.35)), 255, cv2.THRESH_BINARY
            )

        stroke = cv2.bitwise_and(stroke, poly_mask)
        if np.count_nonzero(stroke) < 0.03 * np.count_nonzero(poly_mask):
            stroke = poly_mask

        k_sz = max(7, min(19, int(round(max(h, w) * 0.04)) | 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
        return cv2.dilate(stroke, kernel, iterations=1)

    @staticmethod
    def _check_uniform_background(
        crop_bgr: np.ndarray, text_mask: np.ndarray, global_mask_crop: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        """Return (is_uniform, median_bgr) by inspecting clean pixels around the text mask."""
        outer = cv2.dilate(text_mask, np.ones((15, 15), np.uint8))
        ring = (outer > 0) & (text_mask == 0) & (global_mask_crop == 0)
        pixels = crop_bgr[ring]
        if pixels.shape[0] < 20:
            ring = (outer > 0) & (text_mask == 0)
            pixels = crop_bgr[ring]
        if pixels.shape[0] < 16:
            return False, np.array([255, 255, 255], dtype=np.uint8)

        med = np.median(pixels, axis=0)
        std = float(np.mean(np.std(pixels, axis=0)))
        dists = np.linalg.norm(pixels.astype(np.float32) - med.astype(np.float32), axis=1)
        inlier_ratio = float(np.mean(dists < 24.0))
        lum = float(np.mean(med))
        if (lum >= 235.0 or lum <= 20.0) and inlier_ratio >= 0.72:
            return True, med.astype(np.uint8)
        is_uniform = std < UNIFORM_STD_THRESH and inlier_ratio >= UNIFORM_INLIER_RATIO
        return is_uniform, med.astype(np.uint8)

    def prepare_inpaint(
        self,
        image_bgr: np.ndarray,
        polygons: List[np.ndarray],
        inpaint_mode: str = "auto",
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, int]]:
        """CPU-only mask extraction & uniform solid fills; returns (out_bgr, aot_mask_or_None, stats)."""
        if inpaint_mode not in {"auto", "solid_only", "aot_only"}:
            raise ValueError(f"Unsupported inpaint_mode: {inpaint_mode}")
        h, w = image_bgr.shape[:2]
        out = image_bgr.copy()
        aot_mask = np.zeros((h, w), dtype=np.uint8)
        global_poly_mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            cv2.fillPoly(global_poly_mask, [np.asarray(poly, dtype=np.int32)], 255)

        solid_count = 0
        aot_count = 0

        pad = max(10, int(round(max(h, w) * 0.006)))
        for poly in polygons:
            pts = np.asarray(poly, dtype=np.float32)
            x0 = max(0, int(np.floor(pts[:, 0].min())) - pad)
            y0 = max(0, int(np.floor(pts[:, 1].min())) - pad)
            x1 = min(w, int(np.ceil(pts[:, 0].max())) + pad)
            y1 = min(h, int(np.ceil(pts[:, 1].max())) + pad)
            if x1 <= x0 + 2 or y1 <= y0 + 2:
                continue

            crop = out[y0:y1, x0:x1]
            poly_local = pts - np.array([[x0, y0]], dtype=np.float32)
            text_mask = self._extract_region_mask(crop, poly_local)
            is_uniform, med_bgr = self._check_uniform_background(
                crop, text_mask, global_poly_mask[y0:y1, x0:x1]
            )

            if inpaint_mode == "solid_only" or (inpaint_mode == "auto" and is_uniform):
                solid_count += 1
                crop[text_mask > 0] = med_bgr
            else:
                aot_count += 1
                d_mask = cv2.dilate(text_mask, np.ones((5, 5), np.uint8))
                aot_mask[y0:y1, x0:x1] = np.maximum(aot_mask[y0:y1, x0:x1], d_mask)

        final_mask = aot_mask if (aot_count > 0 and np.any(aot_mask > 0)) else None
        return out, final_mask, {"solid_filled": solid_count, "aot_inpainted": aot_count}

    def finish_aot(self, out: np.ndarray, aot_mask: np.ndarray) -> np.ndarray:
        """GPU/DML AOT-GAN inpainting + feathered composite inside tight ROI."""
        h, w = out.shape[:2]
        rx0, ry0, rx1, ry1 = self._compute_aot_roi(w, h, aot_mask)
        out_roi = out[ry0:ry1, rx0:rx1]
        mask_roi = aot_mask[ry0:ry1, rx0:rx1]
        inpainted_roi = self._run_aot_inpaint_core(out_roi, mask_roi)
        alpha = (cv2.GaussianBlur(mask_roi, (5, 5), 0).astype(np.float32) / 255.0)[..., None]
        alpha = np.where(mask_roi[..., None] > 0, np.maximum(alpha, 0.9), alpha)
        out[ry0:ry1, rx0:rx1] = (
            inpainted_roi.astype(np.float32) * alpha + out_roi.astype(np.float32) * (1.0 - alpha)
        ).clip(0, 255).astype(np.uint8)
        return out

    def remove_text(
        self,
        image_bgr: np.ndarray,
        polygons: List[np.ndarray],
        inpaint_mode: str = "auto",
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Smartly remove text inside all polygons.
        - inpaint_mode="auto": direct solid-color fill for uniform bubbles + mayocream/aot-inpainting for complex artwork
        - inpaint_mode="solid_only": fast solid-color fill only (never loads AOT-GAN weights)
        - inpaint_mode="aot_only": force AOT-GAN inpainting on all text regions
        Returns (cleaned_bgr, stats_dict).
        """
        out, aot_mask, stats = self.prepare_inpaint(image_bgr, polygons, inpaint_mode=inpaint_mode)
        if aot_mask is not None:
            self.finish_aot(out, aot_mask)
        return out, stats

    @staticmethod
    def _compute_aot_roi(w: int, h: int, mask_u8: np.ndarray) -> Tuple[int, int, int, int]:
        pts = cv2.findNonZero(mask_u8)
        if pts is None:
            return 0, 0, w, h
        bx, by, bw, bh = cv2.boundingRect(pts)
        margin = max(96, int(0.10 * max(h, w)))
        rx0 = max(0, bx - margin)
        ry0 = max(0, by - margin)
        rx1 = min(w, bx + bw + margin)
        ry1 = min(h, by + bh + margin)
        min_side = min(256, w, h)
        if (rx1 - rx0) < min_side:
            cx = (rx0 + rx1) // 2
            rx0 = max(0, min(w - min_side, cx - min_side // 2))
            rx1 = min(w, rx0 + min_side)
        if (ry1 - ry0) < min_side:
            cy = (ry0 + ry1) // 2
            ry0 = max(0, min(h - min_side, cy - min_side // 2))
            ry1 = min(h, ry0 + min_side)
        if (rx1 - rx0) * (ry1 - ry0) >= 0.65 * (w * h):
            return 0, 0, w, h
        return rx0, ry0, rx1, ry1

    def _run_aot_inpaint(self, img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        rx0, ry0, rx1, ry1 = self._compute_aot_roi(w, h, mask_u8)
        if (rx0, ry0, rx1, ry1) == (0, 0, w, h):
            return self._run_aot_inpaint_core(img_bgr, mask_u8)
        out = img_bgr.copy()
        out[ry0:ry1, rx0:rx1] = self._run_aot_inpaint_core(
            img_bgr[ry0:ry1, rx0:rx1], mask_u8[ry0:ry1, rx0:rx1]
        )
        return out

    @torch.inference_mode()
    def _run_aot_inpaint_core(self, img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        scale = min(1.0, AOT_MAX_SIDE / max(h, w))
        if scale < 1.0:
            nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
            img_s = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            mask_s = cv2.resize(mask_u8, (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            nh, nw = h, w
            img_s = img_bgr
            mask_s = mask_u8

        target_h = max(80, ((nh + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple)
        target_w = max(80, ((nw + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple)
        pad_h = target_h - nh
        pad_w = target_w - nw
        if pad_h or pad_w:
            img_s = cv2.copyMakeBorder(img_s, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask_s = cv2.copyMakeBorder(mask_s, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        m = (mask_s > 127).astype(np.float32)
        rgb_masked = rgb * (1.0 - m[..., None])

        inp = np.concatenate([m[..., None], rgb_masked], axis=2)  # (H, W, 4): mask + RGB
        self.ensure_loaded()
        if self._ort_session is not None:
            try:
                ort_in = np.transpose(inp, (2, 0, 1))[None, ...].astype(np.float32)
                ort_out = self._ort_session.run(None, {self._ort_input_name: ort_in})[0]
                pred = np.transpose(np.clip(ort_out[0], -1.0, 1.0), (1, 2, 0))
            except Exception as exc:
                print(f"[SmartInpainter] DirectML inference failed, falling back to CPU: {exc}")
                self._ort_session = None
                self._use_ort_dml = False
                self.device_label = "cpu"
                tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)
                pred = self.model(tensor)[0].float().clamp(-1.0, 1.0).cpu().permute(1, 2, 0).numpy()
        else:
            tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)
            if self.use_fp16:
                tensor = tensor.half()
            pred = self.model(tensor)[0].float().clamp(-1.0, 1.0).cpu().permute(1, 2, 0).numpy()
        pred_rgb = ((pred + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_rgb[:nh, :nw], cv2.COLOR_RGB2BGR)

        if scale < 1.0:
            pred_bgr = cv2.resize(pred_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        return pred_bgr
