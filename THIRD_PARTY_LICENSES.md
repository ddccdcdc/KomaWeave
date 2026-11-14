# Third-Party Licenses & Attributions

KomaWeave source code is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** (see [LICENSE](LICENSE)). The following open-source models, libraries, fonts, UI assets, and sample artworks are included or used under their respective licenses:

## Models

- **mayocream/aot-inpainting** (`mayocream/aot-inpainting` on Hugging Face)
  - **License:** MIT License
  - **Notes:** SafeTensors conversion of the AOT-GAN inpainting generator (originally from `zyddnys/manga-image-translator` / `AOT-GAN-for-Inpainting`).

- **RapidOCR & PaddleOCR PP-OCRv6** (`RapidAI/RapidOCR`)
  - **License:** Apache License 2.0
  - **Notes:** General-purpose multilingual text detection, classification, and recognition models.

## Web UI & Fonts

- **Pico CSS v2.1.1** (`static/vendor/pico.min.css`) — MIT License (`https://picocss.com`)
- **Google Fonts** (`Comic Neue`, `Bangers`, `Almarai`, `Lalezar`, `Noto Sans`) — SIL Open Font License 1.1 (OFL-1.1)

## Documentation Showcase Images (`docs/examples/`)

- **Pepper&Carrot Episode 1: "The Potion of Flight" (`pepper_1`–`pepper_2`, Arabic & Chinese Simplified translations)**
  - **Author:** David Revoy (`https://www.peppercarrot.com`)
  - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
  - **Notes:** Translated Arabic and Chinese Simplified derivative images are adapted from the original Episode 1 pages with attribution to David Revoy.

## Core Python Dependencies

- FastAPI, Uvicorn, HTTPX, Pillow, NumPy, OpenCV, PyTorch, ONNX Runtime, Safetensors, Hugging Face Hub, arabic-reshaper, python-bidi (see each package's upstream repository for full license text).
