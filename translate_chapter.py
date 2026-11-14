"""High-speed batch/chapter manga & manhwa translator CLI."""

from __future__ import annotations

import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2

from app.pipeline import MangaTranslationPipeline


async def translate_chapter(
    input_dir: Path,
    output_dir: Path,
    target_lang: str = "ar",
    source_lang: str = "auto",
    ocr_workers: int = 6,
    provider: str = "google_free",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    consolidated: bool = True,
) -> None:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not files:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} images in {input_dir}")
    print(f"Output folder: {output_dir} (target={target_lang}, provider={provider})")

    pipeline = MangaTranslationPipeline()
    t_warm = time.perf_counter()
    pipeline.warmup()
    print(f"Pipeline warmed up in {time.perf_counter() - t_warm:.2f}s (device={pipeline.inpainter.device})")

    t_io0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        images = list(ex.map(lambda p: cv2.imread(str(p)), files))
    valid_pairs = [(f, img) for f, img in zip(files, images) if img is not None]
    if not valid_pairs:
        raise RuntimeError(f"Failed to decode any images in {input_dir}")
    files = [f for f, _ in valid_pairs]
    images = [img for _, img in valid_pairs]
    t_read = (time.perf_counter() - t_io0) * 1000.0
    print(f"Loaded {len(images)} images in {t_read:.1f} ms")

    results, timings = await pipeline.process_chapter(
        images,
        target_lang=target_lang,
        source_lang=source_lang,
        ocr_workers=ocr_workers,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        consolidated=consolidated,
    )

    t_w0 = time.perf_counter()

    def _save_one(item: tuple[Path, object]) -> None:
        src_path, res = item
        out_path = output_dir / src_path.name
        if src_path.suffix.lower() in {".jpg", ".jpeg"}:
            cv2.imwrite(str(out_path), res.image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(str(out_path), res.image_bgr)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_save_one, zip(files, results)))
    t_write = (time.perf_counter() - t_w0) * 1000.0

    total_blocks = sum(len(r.blocks) for r in results)
    total_solid = sum(r.inpaint_stats.get("solid_filled", 0) for r in results)
    total_aot = sum(r.inpaint_stats.get("aot_inpainted", 0) for r in results)

    print("\n=== Chapter Translation Telemetry (Seconds) ===")
    print(f"Pages processed : {len(results)}")
    print(f"Text blocks     : {total_blocks}")
    print(f"Solid-fill boxes: {total_solid} | AOT-GAN inpainted: {total_aot}")
    print(f"OCR total       : {timings['ocr'] / 1000.0:.2f} s ({timings['ocr'] / len(results) / 1000.0:.3f} s/img)")
    print(f"Translate total : {timings['translate'] / 1000.0:.2f} s (parallel with inpaint)")
    print(f"Inpaint total   : {timings['inpaint'] / 1000.0:.2f} s")
    print(f"Render total    : {timings['render'] / 1000.0:.2f} s")
    print(f"Pipeline total  : {timings['total'] / 1000.0:.2f} s ({timings['per_image_avg'] / 1000.0:.3f} s/img)")
    print(f"Disk write      : {t_write / 1000.0:.2f} s")

    await pipeline.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate a folder of manga/manhwa pages in parallel."
    )
    parser.add_argument("input", type=Path, help="Input directory containing chapter images.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (defaults to <input>_<target>).",
    )
    parser.add_argument("--target", default="ar", help="Target language code (default: ar).")
    parser.add_argument("--source", default="auto", help="Source language code (default: auto).")
    parser.add_argument("--workers", type=int, default=6, help="Parallel CPU OCR rec workers.")
    parser.add_argument("--provider", default="google_free", help="Translation provider ID.")
    parser.add_argument("--model", default=None, help="AI model name (optional).")
    parser.add_argument("--api-key", default=None, help="Provider API key (optional).")
    parser.add_argument("--base-url", default=None, help="Custom OpenAI-compatible base URL.")
    parser.add_argument(
        "--no-consolidated",
        action="store_true",
        help="Translate each image independently instead of consolidating the chapter.",
    )
    args = parser.parse_args()

    out_dir = args.output or args.input.parent / f"{args.input.name}_{args.target}"
    asyncio.run(
        translate_chapter(
            input_dir=args.input,
            output_dir=out_dir,
            target_lang=args.target,
            source_lang=args.source,
            ocr_workers=args.workers,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            consolidated=not args.no_consolidated,
        )
    )


if __name__ == "__main__":
    main()
