# Privacy & Data Handling Policy

KomaWeave runs **OCR, image inpainting, and typesetting 100% locally** on your machine. However, **text translation** can send extracted text to external services depending on the provider you select.

## 1. What Data Leaves Your Device
- **Images:** Never sent to any external translation provider by KomaWeave. Image decoding, OCR, inpainting, and typesetting occur strictly in local memory.
- **Extracted Text:**
  - **`google_free` (Default):** Sends extracted text strings over HTTPS to Google Translate's public web translation endpoints (`clients5.google.com` / `translate.googleapis.com`), **not** the paid Google Cloud Translation API.
  - **Cloud AI Providers (`openai`, `gemini`, `deepseek`, `anthropic`, `groq`, `openrouter`):** Sends extracted text strings and translation system prompts to the selected provider's API using your configured API key.
  - **Local / Self-Hosted (`custom` with local Ollama/vLLM):** When configured with a local endpoint (`KOMAWEAVE_ALLOW_LOCAL_PROVIDERS=1`), **zero data leaves your machine**.
  - **OCR-only mode (`mode="ocr"` / `POST /api/ocr`):** **Zero data leaves your machine** (no translation call is made).

## 2. Retention & Storage
- **HTTP Server (`/api/translate`, `/api/translate-batch`, `/api/ocr`):** Uploaded images and translated outputs are processed in memory and returned in the HTTP response; KomaWeave does **not** save uploaded images or translated pages to disk.
- **CLI (`translate_chapter.py`):** Writes translated images only to the local output folder you specify.
- **Provider API Keys:** When saved via the Web UI, API keys are stored locally in `.secrets.json` (git-ignored, `0600` permissions where supported by the OS).
- **Third-Party Retention:** Data retention by Google Translate or cloud AI providers is governed by each third-party provider's own privacy policy and terms of service.

## 3. Confidential or Unpublished Content
Do **not** use `google_free` or third-party cloud AI providers for confidential, NDA-protected, or unpublished manuscripts unless you accept that provider's data-handling terms. For private/offline workflows, use a local LLM via `custom` (`KOMAWEAVE_ALLOW_LOCAL_PROVIDERS=1`) or OCR-only mode.
