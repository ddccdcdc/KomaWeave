"""Launch the KomaWeave web server (auto-entering .venv and bootstrapping setup if needed)."""

from __future__ import annotations

import argparse
import os
import platform
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _bootstrap_venv() -> None:
    """Automatically enter .venv (and run setup_project.py if dependencies are missing)."""
    if os.getenv("KOMAWEAVE_NO_VENV_REEXEC") == "1":
        return

    vpy = _venv_python()
    in_local_venv = Path(sys.prefix).resolve() == VENV_DIR.resolve()

    if not in_local_venv and vpy.exists():
        # Verify core package inside .venv; run setup if incomplete
        chk = subprocess.run(
            [str(vpy), "-c", "import uvicorn, fastapi, rapidocr, torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if chk.returncode != 0:
            print("[KomaWeave] Completing environment setup inside .venv ...")
            subprocess.check_call([str(vpy), str(ROOT / "setup_project.py")])
        raise SystemExit(subprocess.call([str(vpy), str(Path(__file__).resolve()), *sys.argv[1:]]))

    # Already inside a venv or system Python without .venv: check imports
    try:
        import fastapi  # noqa: F401
        import rapidocr  # noqa: F401
        import torch  # noqa: F401
        import uvicorn  # noqa: F401
        return
    except ImportError:
        pass

    if not in_local_venv:
        print("[KomaWeave] Environment not configured yet; running setup_project.py --venv ...")
        subprocess.check_call([sys.executable, str(ROOT / "setup_project.py"), "--venv"])
        if vpy.exists():
            raise SystemExit(subprocess.call([str(vpy), str(Path(__file__).resolve()), *sys.argv[1:]]))
    else:
        print("[KomaWeave] Installing missing dependencies into .venv ...")
        subprocess.check_call([sys.executable, str(ROOT / "setup_project.py")])


_bootstrap_venv()

import uvicorn  # noqa: E402

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the KomaWeave server.")
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "127.0.0.1"),
        help="Bind host (use 0.0.0.0 to expose on LAN/external clients). Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Bind port. Default: 8000",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Allow running on a non-loopback host without KOMAWEAVE_API_KEY (not recommended).",
    )
    args = parser.parse_args()

    is_loopback = args.host.strip().lower() in _LOOPBACK_HOSTS
    if not is_loopback and not os.getenv("KOMAWEAVE_API_KEY", "").strip():
        if not args.allow_unauthenticated:
            token = secrets.token_urlsafe(32)
            os.environ["KOMAWEAVE_API_KEY"] = token
            print(
                "\n[KomaWeave Security] Non-loopback bind detected without KOMAWEAVE_API_KEY.\n"
                f"[KomaWeave Security] Generated session API key: {token}\n"
                "[KomaWeave Security] Pass this key in the Web UI or via 'X-API-Key' / 'Authorization: Bearer'.\n"
            )
        else:
            print(
                "\n[KomaWeave WARNING] Running unauthenticated on a non-loopback host! "
                "Anyone on the network can access OCR, GPU, and provider endpoints.\n"
            )

    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
