# Third-Party Licenses & Attributions

KomaWeave source code is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** (see [LICENSE](LICENSE)). The following third-party models, libraries, fonts, UI assets, and sample artworks are included or used under their respective licenses:

---

## Models

- **mayocream/aot-inpainting** (`https://huggingface.co/mayocream/aot-inpainting`)
  - **License:** MIT License
  - **Notes:** SafeTensors conversion of the AOT-GAN inpainting generator, originally derived from `researchmm/AOT-GAN-for-Inpainting` (Apache License 2.0) and `zyddnys/manga-image-translator` (GNU General Public License v3.0).

- **RapidOCR & PaddleOCR PP-OCRv6** (`https://github.com/RapidAI/RapidOCR`, `https://github.com/PaddlePaddle/PaddleOCR`)
  - **License:** Apache License 2.0
  - **Notes:** Multilingual text detection, orientation classification, and recognition engines and weights.

---

## Web UI & Fonts

- **Pico CSS v2.1.1** (`static/vendor/pico.min.css`, `https://picocss.com`)
  - **Copyright:** `Copyright 2019-2025 Pico CSS`
  - **License:** MIT License (header preserved in `static/vendor/pico.min.css`)

- **Bundled Fonts (`fonts/` directory)** — **SIL Open Font License, Version 1.1 (OFL-1.1)**
  Full license text is included at [fonts/OFL.txt](fonts/OFL.txt).
  - `fonts/Arabic-Dialogue.ttf` (**Almarai**) — `Copyright (c) 2019 by Almarai. All rights reserved.` (`https://github.com/google/fonts/tree/main/ofl/almarai`)
  - `fonts/Arabic-SFX.ttf` (**Lalezar**) — `Copyright 2015 The Lalezar Project Authors (https://github.com/BornaIz/Lalezar)`
  - `fonts/MangaLatin-Bold.ttf` (**Comic Neue**) — `Copyright 2014 The Comic Neue Project Authors (https://github.com/crozynski/comicneue)`
  - `fonts/MangaSFX-Bold.ttf` (**Bangers**) — `Copyright 2010 The Bangers Project Authors (https://github.com/googlefonts/bangers)`
  - `fonts/NotoSans-Bold.ttf` (**Noto Sans**) — `Copyright 2022 The Noto Project Authors (https://github.com/notofonts/latin-greek-cyrillic)`

---

## Documentation Showcase Images (`docs/examples/`)

- **Pepper&Carrot Episode 1: "The Potion of Flight" (`pepper_1.jpg`, `pepper_2.jpg`, and translated variants)**
  - **Author:** David Revoy (`https://www.peppercarrot.com`)
  - **License:** Creative Commons Attribution 4.0 International (CC BY 4.0 — `https://creativecommons.org/licenses/by/4.0/`)
  - **Changes Made:** `pepper_1_ar.jpg`, `pepper_1_zh.jpg`, `pepper_2_ar.jpg`, and `pepper_2_zh.jpg` are derivative translations into Arabic and Chinese Simplified generated and typeset with KomaWeave.

---

## Python Dependencies

| Package | License | Upstream |
| --- | --- | --- |
| `fastapi` | MIT | `https://github.com/fastapi/fastapi` |
| `uvicorn` | BSD-3-Clause | `https://github.com/encode/uvicorn` |
| `python-multipart` | Apache-2.0 | `https://github.com/Kludex/python-multipart` |
| `httpx` | BSD-3-Clause | `https://github.com/encode/httpx` |
| `numpy` | BSD-3-Clause | `https://github.com/numpy/numpy` |
| `opencv-python-headless` | Apache-2.0 (OpenCV) / MIT (packaging wrapper) | `https://github.com/opencv/opencv-python` |
| `pillow` | HPND (Historical Permission Notice and Disclaimer) | `https://github.com/python-pillow/Pillow` |
| `rapidocr` | Apache-2.0 | `https://github.com/RapidAI/RapidOCR` |
| `safetensors` | Apache-2.0 | `https://github.com/huggingface/safetensors` |
| `huggingface-hub` | Apache-2.0 | `https://github.com/huggingface/huggingface_hub` |
| `arabic-reshaper` | MIT | `https://github.com/mpcabd/python-arabic-reshaper` |
| `python-bidi` | LGPL-3.0-only | `https://github.com/MeirKriheli/python-bidi` |
| `torch` | BSD-3-Clause | `https://github.com/pytorch/pytorch` |
| `onnxruntime` / `onnxruntime-gpu` | MIT | `https://github.com/microsoft/onnxruntime` |
