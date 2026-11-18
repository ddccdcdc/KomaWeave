"""Launch the KomaWeave web server (auto-entering .venv and bootstrapping setup if needed)."""

from __future__ import annotations

import argparse
import os
import secrets

from setup_project import ensure_runtime_env

ensure_runtime_env(("uvicorn", "fastapi", "cv2", "rapidocr", "torch"))

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
