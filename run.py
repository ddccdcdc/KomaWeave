"""Launch the KomaWeave web server (auto-entering .venv and bootstrapping setup if needed)."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import urllib.request

from setup_project import ensure_runtime_env

ensure_runtime_env(("uvicorn", "fastapi", "cv2", "rapidocr", "torch"))

import uvicorn  # noqa: E402

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_ROOT = Path(__file__).resolve().parent


def _parse_ver(v: str):
    try:
        from packaging.version import Version

        return Version(v)
    except Exception:
        return tuple(int(p) if p.isdigit() else 0 for p in v.split(".")[:3])


def _check_repo_update() -> tuple[int, str]:
    """Return (commits_behind, latest_subject) if origin/main is ahead of HEAD."""
    if not (_ROOT / ".git").exists() or not shutil.which("git"):
        return 0, ""
    try:
        subprocess.run(
            ["git", "-C", str(_ROOT), "fetch", "origin", "main", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=4,
            check=False,
        )
        cnt_res = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-list", "--count", "HEAD..origin/main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )
        behind = int(cnt_res.stdout.strip() or "0") if cnt_res.returncode == 0 else 0
        if behind <= 0:
            return 0, ""
        log_res = subprocess.run(
            ["git", "-C", str(_ROOT), "log", "-1", "--pretty=%h %s", "origin/main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )
        return behind, log_res.stdout.strip()
    except Exception:
        return 0, ""


def _check_pkg_updates() -> list[tuple[str, str, str]]:
    """Return [(pkg_name, installed_ver, latest_ver)] for outdated core packages."""
    candidates = ["rapidocr", "onnx", "huggingface-hub"]
    for ort_pkg in ("onnxruntime-directml", "onnxruntime-gpu", "onnxruntime"):
        try:
            importlib.metadata.version(ort_pkg)
            candidates.append(ort_pkg)
            break
        except importlib.metadata.PackageNotFoundError:
            continue

    outdated: list[tuple[str, str, str]] = []
    for pkg in candidates:
        try:
            cur = importlib.metadata.version(pkg)
            req = urllib.request.Request(
                f"https://pypi.org/pypi/{pkg}/json",
                headers={"User-Agent": "KomaWeave-Updater"},
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                latest = json.loads(resp.read().decode("utf-8"))["info"]["version"]
            if _parse_ver(latest) > _parse_ver(cur):
                outdated.append((pkg, cur, latest))
        except Exception:
            continue
    return outdated


def _check_and_prompt_updates() -> None:
    if os.getenv("KOMAWEAVE_SKIP_UPDATE_CHECK") == "1":
        return
    behind, latest_msg = _check_repo_update()
    outdated_pkgs = _check_pkg_updates()
    if behind <= 0 and not outdated_pkgs:
        return

    print("\n[KomaWeave Update Available]")
    if behind > 0:
        print(f"  * KomaWeave repository: {behind} new commit(s) on main ({latest_msg})")
    for pkg, cur, latest in outdated_pkgs:
        print(f"  * Library update: {pkg} {cur} -> {latest}")

    if not sys.stdin.isatty():
        print("  Run 'git pull origin main' or 'pip install -U ...' to update.\n")
        return

    try:
        ans = input("Install updates before starting the server? [y/N]: ").strip().lower()
    except EOFError:
        return
    if ans not in {"y", "yes"}:
        return

    updated = False
    if behind > 0:
        res = subprocess.run(["git", "-C", str(_ROOT), "pull", "--ff-only", "origin", "main"], check=False)
        if res.returncode == 0:
            updated = True
        else:
            print("[KomaWeave Update] Could not fast-forward git pull (local changes?). Continuing...")
    if outdated_pkgs:
        pkgs = [pkg for pkg, _, _ in outdated_pkgs]
        res = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs], check=False)
        if res.returncode == 0:
            updated = True
    if behind > 0 and updated:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(_ROOT / "requirements.txt")],
            check=False,
        )
    if updated:
        print("[KomaWeave Update] Update complete! Restarting KomaWeave...\n")
        raise SystemExit(
            subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--no-update-check"])
        )


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
    parser.add_argument(
        "--no-update-check",
        action="store_true",
        help="Skip checking for KomaWeave and library updates on startup.",
    )
    args = parser.parse_args()

    if not args.no_update_check:
        _check_and_prompt_updates()

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
