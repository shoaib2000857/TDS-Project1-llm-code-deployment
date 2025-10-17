import os, re, pathlib, time
import google.generativeai as genai
from typing import List, Dict

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

def _extract_files_from_markdown(md: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    blocks = re.findall(r"```([a-zA-Z0-9.+-]*)\n(.*?)```", md, flags=re.DOTALL)
    html_count = js_count = css_count = 0
    for lang, content in blocks:
        lang_lower = (lang or "").lower()
        if "html" in lang_lower:
            html_count += 1
            name = "index.html" if html_count == 1 else f"page{html_count}.html"
            files[name] = content.strip()
        elif "javascript" in lang_lower or lang_lower in ("js", "jsx"):
            js_count += 1
            name = "script.js" if js_count == 1 else f"script{js_count}.js"
            files[name] = content.strip()
        elif "css" in lang_lower:
            css_count += 1
            name = "style.css" if css_count == 1 else f"style{css_count}.css"
            files[name] = content.strip()
    if not files:
        files["index.html"] = md.strip()
    return files

def _prompt_for_webapp(brief: str, attachment_names: List[str]) -> str:
    import json
    return f"""
You are an expert frontend engineer. Rebuild a clean, production-ready static web app for the following brief:

BRIEF:
{brief}

ATTACHMENTS (relative URLs):
{json.dumps(attachment_names)}

REQUIREMENTS:
- Output only fenced code blocks:
  - ```html``` main HTML (with <head> + title)
  - ```css``` for styling
  - ```javascript``` for logic
- Fully static. Must run on GitHub Pages.
- If attachments include data, load via fetch('filename').
- Keep design simple, accessible, and elegant.
- Avoid commentary or markdown outside code blocks.
"""

def _write_minimal_fallback(work_dir: str, brief: str):
    root = pathlib.Path(work_dir)
    (root / "index.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Auto App</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<link rel="stylesheet" href="style.css" />
</head>
<body>
<main>
  <h1>Auto App</h1>
  <p>This minimal page was generated as a fallback.</p>
  <pre id="brief">{brief}</pre>
</main>
<script src="script.js"></script>
</body>
</html>
""", encoding="utf-8")
    (root / "style.css").write_text(
        "body{font-family:system-ui,Arial,sans-serif;padding:2rem}pre{white-space:pre-wrap}",
        encoding="utf-8",
    )
    (root / "script.js").write_text("console.log('fallback app ready');", encoding="utf-8")

def generate_app_with_gemini(brief: str, work_dir: str, attachment_names: List[str]) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY not set — writing fallback app.")
        _write_minimal_fallback(work_dir, brief)
        return

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(DEFAULT_MODEL)

        prompt = _prompt_for_webapp(brief, attachment_names)
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=4096,
            ),
        )

        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            print("⚠️ Gemini returned empty — writing fallback app.")
            _write_minimal_fallback(work_dir, brief)
            return

        files = _extract_files_from_markdown(text)
        if "index.html" not in files:
            first = next(iter(files.values()))
            files["index.html"] = first

        root = pathlib.Path(work_dir)
        for name, content in files.items():
            (root / name).write_text(content, encoding="utf-8")

        print(f"✅ Gemini generated: {list(files.keys())}")

    except Exception as e:
        print(f"⚠️ Gemini error: {e} — writing fallback app.")
        _write_minimal_fallback(work_dir, brief)
