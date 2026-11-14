"""Single-command reproducible setup for the Manga & Manhwa Translation System."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FONTS_DIR = ROOT / "fonts"
VENV_DIR = ROOT / ".venv"

FONT_URLS = {
    "MangaLatin-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/comicneue/ComicNeue-Bold.ttf",
    "MangaSFX-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/bangers/Bangers-Regular.ttf",
    "Arabic-Dialogue.ttf": "https://github.com/google/fonts/raw/main/ofl/almarai/Almarai-ExtraBold.ttf",
    "Arabic-SFX.ttf": "https://github.com/google/fonts/raw/main/ofl/lalezar/Lalezar-Regular.ttf",
    "NotoSans-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
}


def _in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _ensure_venv_and_reexec() -> None:
    if _in_virtualenv():
        return
    print(f"[0/4] Creating virtual environment at {VENV_DIR} ...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    if platform.system() == "Windows":
        vpy = VENV_DIR / "Scripts" / "python.exe"
    else:
        vpy = VENV_DIR / "bin" / "python"
    args = [str(vpy), str(Path(__file__).resolve()), *[a for a in sys.argv[1:] if a != "--venv"]]
    raise SystemExit(subprocess.call(args))


def _detect_accelerator(requested: str) -> str:
    if requested != "auto":
        return requested
    system = platform.system()
    if system in {"Windows", "Linux"} and shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                check=False,
            )
            if res.returncode == 0 and "GPU" in res.stdout:
                return "cuda"
        except Exception:
            pass
    if system == "Darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def _pip_install(packages: list[str], extra_index_url: str | None = None) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    if extra_index_url:
        cmd.extend(["--index-url", extra_index_url])
    subprocess.check_call(cmd)


def install_dependencies(accel: str) -> None:
    print(f"[1/4] Detected hardware mode: {accel.upper()}")
    print("[2/4] Installing PyTorch, ONNX Runtime, and project dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    if accel == "cuda":
        # Check if CUDA-enabled torch is already installed
        has_cuda_torch = False
        try:
            import torch  # type: ignore

            has_cuda_torch = bool(torch.cuda.is_available())
        except Exception:
            pass
        if not has_cuda_torch:
            _pip_install(["torch>=2.5.0"], extra_index_url="https://download.pytorch.org/whl/cu124")
        _pip_install(["onnxruntime-gpu>=1.20.0,<1.24.0"])
    else:
        _pip_install(["torch>=2.5.0", "onnxruntime>=1.20.0"])

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]
    )


def download_fonts() -> None:
    print("[3/4] Ensuring manga & multilingual fonts are present...")
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FONT_URLS.items():
        dest = FONTS_DIR / name
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"  - {name}: already present")
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"  - {name}: downloaded ({len(data):,} bytes)")


def precache_models() -> None:
    print("[4/4] Pre-caching OCR (PP-OCRv6) & inpainting (mayocream/aot-inpainting) models...")
    sys.path.insert(0, str(ROOT))
    from app.inpainter import SmartInpainter
    from app.ocr_engine import GeneralOCREngine

    ocr = GeneralOCREngine()
    inp = SmartInpainter()
    print(
        f"Setup complete! (OCR CUDA={ocr.use_cuda}, Inpainter device={inp.device})\n"
        "Start the server with: python run.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command setup for Manga & Manhwa Translator.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Target accelerator (default: auto-detect NVIDIA CUDA / Apple MPS / CPU).",
    )
    parser.add_argument(
        "--venv",
        action="store_true",
        help="Create and install into a local .venv directory automatically.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip pre-downloading OCR and inpainting weights.",
    )
    args = parser.parse_args()

    if args.venv:
        _ensure_venv_and_reexec()

    accel = _detect_accelerator(args.device)
    install_dependencies(accel)
    download_fonts()
    if not args.skip_models:
        precache_models()


if __name__ == "__main__":
    main()
