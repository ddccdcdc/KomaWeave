import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FONTS_DIR = BASE_DIR / "fonts"
STATIC_DIR = BASE_DIR / "static"

# Supported target languages (code -> display name)
LANGUAGES = {
    "ar": "Arabic (العربية)",
    "en": "English",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "it": "Italian (Italiano)",
    "pt": "Portuguese (Português)",
    "ru": "Russian (Русский)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "zh-CN": "Chinese Simplified (简体中文)",
    "zh-TW": "Chinese Traditional (繁體中文)",
    "tr": "Turkish (Türkçe)",
    "id": "Indonesian (Bahasa Indonesia)",
    "vi": "Vietnamese (Tiếng Việt)",
    "th": "Thai (ไทย)",
    "hi": "Hindi (हिन्दी)",
    "pl": "Polish (Polski)",
    "nl": "Dutch (Nederlands)",
    "uk": "Ukrainian (Українська)",
    "fa": "Persian (فارسی)",
    "ur": "Urdu (اردو)",
}

RTL_LANGUAGES = {"ar", "he", "fa", "ur", "ps", "sd", "ug"}

# Font paths (downloaded by setup_project.py if missing)
FONT_LATIN = FONTS_DIR / "MangaLatin-Bold.ttf"
FONT_SFX = FONTS_DIR / "MangaSFX-Bold.ttf"
FONT_ARABIC = FONTS_DIR / "Arabic-Dialogue.ttf"
FONT_ARABIC_SFX = FONTS_DIR / "Arabic-SFX.ttf"
FONT_FALLBACK = FONTS_DIR / "NotoSans-Bold.ttf"

# OCR settings (general-purpose PP-OCRv6)
OCR_DET_LIMIT_SIDE = int(os.getenv("OCR_DET_LIMIT_SIDE", "1088"))
OCR_DET_BOX_THRESH = float(os.getenv("OCR_DET_BOX_THRESH", "0.28"))
OCR_DET_THRESH = float(os.getenv("OCR_DET_THRESH", "0.20"))
OCR_DET_UNCLIP_RATIO = float(os.getenv("OCR_DET_UNCLIP_RATIO", "1.6"))

# Inpainting settings (mayocream/aot-inpainting)
AOT_REPO_ID = "mayocream/aot-inpainting"
AOT_MAX_SIDE = int(os.getenv("AOT_MAX_SIDE", "1024"))

# Smart uniform fill thresholds
UNIFORM_STD_THRESH = 15.0
UNIFORM_INLIER_RATIO = 0.82

# Security, upload & resource limits
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "300"))
MAX_BATCH_TOTAL_BYTES = int(os.getenv("MAX_BATCH_TOTAL_MB", "250")) * 1024 * 1024
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "16384"))
MAX_IMAGE_HEIGHT = int(os.getenv("MAX_IMAGE_HEIGHT", "32768"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "120000000"))
MAX_WORKING_SIDE = int(os.getenv("MAX_WORKING_SIDE", "3600"))
MAX_CONCURRENT_GPU_JOBS = int(os.getenv("MAX_CONCURRENT_GPU_JOBS", "2"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
TRANSLATE_RATE_LIMIT_PER_MINUTE = int(os.getenv("TRANSLATE_RATE_LIMIT_PER_MINUTE", "30"))
MODEL_DISCOVERY_RATE_LIMIT_PER_MINUTE = int(os.getenv("MODEL_DISCOVERY_RATE_LIMIT_PER_MINUTE", "20"))
ALLOW_LOCAL_PROVIDER_URLS = os.getenv("KOMAWEAVE_ALLOW_LOCAL_PROVIDERS", "0") == "1"
