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

- **Wikipe-tan Comic Pages (`wikipe_1`–`wikipe_3` and translated derivatives)**
  - **Author:** Kasuga (Japanese Wikipedia)
  - **License:** Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) / GNU Free Documentation License (GFDL)
  - **Notes:** Translated English and Arabic derivative images are likewise shared under CC BY-SA 3.0 / GFDL.

- **Pepper&Carrot Episode 1: "The Potion of Flight" (`pepper_1`–`pepper_2` and Arabic translations)**
  - **Author:** David Revoy (`https://www.peppercarrot.com`)
  - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
  - **Notes:** Translated Arabic derivative images are adapted from the original Episode 1 pages with attribution to David Revoy.

## Core Python Dependencies

- FastAPI, Uvicorn, HTTPX, Pillow, NumPy, OpenCV, PyTorch, ONNX Runtime, Safetensors, Hugging Face Hub, arabic-reshaper, python-bidi (see each package's upstream repository for full license text).
