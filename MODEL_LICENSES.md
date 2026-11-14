# Model Weights & Licensing

This document covers the licensing and provenance of the **machine-learning model weights** downloaded and executed by KomaWeave at setup/runtime. (For Python libraries, fonts, and UI assets, see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).)

> **Important:** KomaWeave does **not** bundle or redistribute model weight binaries in this Git repository. `setup_project.py` and the runtime pipeline fetch weights directly from their upstream hosts (Hugging Face Hub and RapidOCR's model mirrors) onto the user's machine.

---

## 1. AOT-Inpainting Generator (`mayocream/aot-inpainting`)

| Field | Value |
| --- | --- |
| **Artifact** | `model.safetensors`, `config.json` |
| **Upstream Host** | `https://huggingface.co/mayocream/aot-inpainting` |
| **Pinned Commit / Revision** | `cffe2346ac2b5ebe1f2d61335d602d12cc144c6f` |
| **SHA-256 (`model.safetensors`)** | `1b4fea17a84a228c2097a42ab2f403357f07bb56ae022dc243b40817b7aa87d1` |
| **Declared Repo License** | **MIT License** |

### Upstream Provenance Chain
1. **`researchmm/AOT-GAN-for-Inpainting`** (`https://github.com/researchmm/AOT-GAN-for-Inpainting`) — **Apache License 2.0** (original AOT-GAN architecture & training codebase).
2. **`zyddnys/manga-image-translator`** (`https://github.com/zyddnys/manga-image-translator`) — **GNU General Public License v3.0 (GPL-3.0)** (manga-domain inpainting checkpoint).
3. **`mayocream/aot-inpainting`** (`https://huggingface.co/mayocream/aot-inpainting`) — **MIT License** (SafeTensors conversion of the AOT generator).

### Legal Justification for Use with KomaWeave
- KomaWeave does not redistribute the checkpoint file in this repository; users fetch it directly from Hugging Face.
- KomaWeave's own codebase is licensed under **GNU AGPL-3.0** ([LICENSE](LICENSE)), which is compatible with **GPL-3.0** (via GPL-3.0 / AGPL-3.0 Section 13 compatibility), **Apache-2.0** (compatible with GPLv3/AGPLv3), and **MIT**.
- KomaWeave's PyTorch `AOTGenerator` wrapper in `app/inpainter.py` was written from scratch for SafeTensors loading without copying GPL-3.0 source code.

---

## 2. OCR Detection, Classification & Recognition (`RapidOCR` / `PaddleOCR`)

| Component | Model Files | Upstream Source | License |
| --- | --- | --- | --- |
| **Text Detection** | `PP-OCRv6_det_tiny.onnx`, `PP-OCRv6_det_medium.onnx` (and `ch_PP-OCRv4` / `PP-OCRv5` variants) | `PaddlePaddle/PaddleOCR` via `RapidAI/RapidOCR` | **Apache License 2.0** |
| **Orientation Classifier** | `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | `PaddlePaddle/PaddleOCR` via `RapidAI/RapidOCR` | **Apache License 2.0** |
| **Text Recognition** | `PP-OCRv6_rec_tiny.onnx`, `PP-OCRv6_rec_mobile.onnx` | `PaddlePaddle/PaddleOCR` via `RapidAI/RapidOCR` | **Apache License 2.0** |

- **Rights Clarification:** `RapidAI/RapidOCR` (`https://github.com/RapidAI/RapidOCR`) provides the ONNX Runtime inference wrapper and model packaging under **Apache-2.0**. The underlying PP-OCR model weights originate from **Baidu's PaddleOCR** (`https://github.com/PaddlePaddle/PaddleOCR`) and are likewise licensed under **Apache-2.0**.
- Apache-2.0 grants permission for commercial use, modification, and distribution provided copyright and license notices are preserved, and does not grant trademark endorsement rights.
