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
