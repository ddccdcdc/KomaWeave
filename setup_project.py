"""Single-command reproducible setup for KomaWeave."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FONTS_DIR = ROOT / "fonts"
VENV_DIR = ROOT / ".venv"

VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

FONT_URLS = {
    "MangaLatin-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/comicneue/ComicNeue-Bold.ttf",
    "MangaSFX-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/bangers/Bangers-Regular.ttf",
    "Arabic-Dialogue.ttf": "https://github.com/google/fonts/raw/main/ofl/almarai/Almarai-ExtraBold.ttf",
    "Arabic-SFX.ttf": "https://github.com/google/fonts/raw/main/ofl/lalezar/Lalezar-Regular.ttf",
    "NotoSans-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
}


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _in_virtualenv() -> bool:
    return Path(sys.prefix).resolve() == VENV_DIR.resolve() or (
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    )


def _ensure_venv_and_reexec() -> None:
    if _in_virtualenv():
        return
    vpy = _venv_python()
    if not vpy.exists():
        print(f"[0/4] Creating virtual environment at {VENV_DIR} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"[0/4] Using existing virtual environment at {VENV_DIR} ...")
    args = [str(vpy), str(Path(__file__).resolve()), *[a for a in sys.argv[1:] if a != "--venv"]]
    raise SystemExit(subprocess.call(args))


def _ensure_windows_vc_redist() -> None:
    """Ensure Microsoft VC++ 2015-2022 Redistributable DLLs exist on Windows."""
    if platform.system() != "Windows":
        return
    missing = False
    for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            missing = True
            break
    if not missing:
        return
    print("[0/4] Installing missing Microsoft Visual C++ Redistributable (x64)...")
    installer = Path(tempfile.gettempdir()) / "vc_redist.x64.exe"
    try:
        req = urllib.request.Request(VC_REDIST_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            installer.write_bytes(resp.read())
        res = subprocess.run(
            [str(installer), "/install", "/quiet", "/norestart"],
            check=False,
        )
        # 0 = success, 3010 = reboot required (DLLs are already usable), 1638 = newer version already present
        if res.returncode not in {0, 1638, 3010}:
            print(f"  [Warning] vc_redist installer exited with code {res.returncode}")
    except Exception as exc:
        print(f"  [Warning] Could not auto-install VC++ Redistributable: {exc}")
    finally:
        try:
            if installer.exists():
                installer.unlink()
        except OSError:
            pass


def ensure_runtime_env(required_modules: tuple[str, ...]) -> None:
    """Auto-enter .venv and run setup_project.py if any required runtime dependency is missing."""
    if os.getenv("KOMAWEAVE_NO_VENV_REEXEC") == "1":
        return

    vpy = _venv_python()
    in_local_venv = Path(sys.prefix).resolve() == VENV_DIR.resolve()

    if not in_local_venv and vpy.exists():
        check_expr = "import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in %r) else 1)" % (
            list(required_modules),
        )
        chk = subprocess.run(
            [str(vpy), "-c", check_expr],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if chk.returncode != 0:
            print("[KomaWeave] Completing environment setup inside .venv ...")
            subprocess.check_call([str(vpy), str(Path(__file__).resolve())])
        raise SystemExit(subprocess.call([str(vpy), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]))

    if all(importlib.util.find_spec(m) is not None for m in required_modules):
        return

    if not in_local_venv:
        print("[KomaWeave] Environment not configured yet; running setup_project.py --venv ...")
        subprocess.check_call([sys.executable, str(Path(__file__).resolve()), "--venv"])
        if vpy.exists():
            raise SystemExit(subprocess.call([str(vpy), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]))
    else:
        print("[KomaWeave] Installing missing dependencies into .venv ...")
        subprocess.check_call([sys.executable, str(Path(__file__).resolve())])


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
    if system == "Linux" and (Path("/dev/kfd").exists() or shutil.which("rocm-smi")):
        return "rocm"
    if system == "Windows":
        try:
            ps = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ';'",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=6,
                check=False,
            )
            gpus = ps.stdout.lower()
            if any(k in gpus for k in ("amd", "radeon", "intel", "arc", "iris", "uhd")):
                return "dml"
        except Exception:
            pass
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
        has_cuda_torch = False
        try:
            import torch  # type: ignore

            has_cuda_torch = bool(torch.cuda.is_available())
        except Exception:
            pass
        if not has_cuda_torch:
            _pip_install(["torch>=2.5.0"], extra_index_url="https://download.pytorch.org/whl/cu124")
        _pip_install(["onnxruntime-gpu>=1.20.0,<1.24.0"])
    elif accel == "rocm":
        _pip_install(["torch>=2.5.0"], extra_index_url="https://download.pytorch.org/whl/rocm6.2")
        _pip_install(["onnxruntime>=1.20.0"])
    elif accel == "dml":
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        _pip_install(["torch>=2.5.0", "onnxruntime-directml>=1.20.0"])
        # Optional DirectML PyTorch backend (best-effort; skips cleanly if wheel unavailable for Python version)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "torch-directml"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
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
    import cv2
    import numpy as np

    from app.inpainter import SmartInpainter
    from app.ocr_engine import GeneralOCREngine

    dummy = np.full((320, 320, 3), 255, dtype=np.uint8)
    cv2.putText(dummy, "TEST", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)

    ocr = GeneralOCREngine()
    ocr.extract(dummy)
    ocr.extract_batch([dummy, dummy])

    inp = SmartInpainter()
    inp.remove_text(
        dummy,
        [np.array([[30, 110], [260, 110], [260, 190], [30, 190]], dtype=np.float32)],
        inpaint_mode="aot_only",
    )
    print(
        f"Setup complete! (OCR CUDA={ocr.use_cuda}, OCR DML={ocr.use_dml}, Inpainter device={inp.device_label})\n"
        "Start the server with: python run.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command setup for KomaWeave.")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "dml", "rocm", "mps", "cpu"],
        default="auto",
        help="Target accelerator (default: auto-detect NVIDIA CUDA / Windows DirectML / Linux ROCm / Apple MPS / CPU).",
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

    _ensure_windows_vc_redist()

    if args.venv:
        _ensure_venv_and_reexec()

    accel = _detect_accelerator(args.device)
    install_dependencies(accel)
    download_fonts()
    if not args.skip_models:
        precache_models()


if __name__ == "__main__":
    main()
