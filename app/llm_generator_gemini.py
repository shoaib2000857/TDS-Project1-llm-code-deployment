import os, re, tempfile, pathlib, json
import google.generativeai as genai
from typing import List, Dict

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

# ------------------------------------------------------------
# Utility: extract code blocks -> files
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Build prompt for the model
# ------------------------------------------------------------
def _prompt_for_webapp(brief: str, attachment_names: List[str], existing_code: str = "") -> str:
    """
    Construct a deterministic prompt. Includes old code if present.
    """
    base = f"""
You are an expert frontend engineer. Build or update a small, production-ready static web app.

BRIEF:
{brief}

ATTACHMENTS AVAILABLE IN THE SAME FOLDER (relative URLs):
{json.dumps(attachment_names)}

REQUIREMENTS:
- Output your ENTIRE solution as fenced code blocks only.
  - ```html``` for HTML (must include <head> with meta charset and title)
  - ```css``` for styles (optional)
  - ```javascript``` for logic (optional)
- The app must be fully static (no server code) and work on GitHub Pages.
- If attachments include data files, load them with relative fetch().
- Keep it small, clean, accessible, and self-contained.
"""

    # 🧩 Added for Round-2 awareness
    if existing_code.strip():
        base += f"""

EXISTING PROJECT CODE:
Below is the current app code. Update or extend it intelligently based on the new brief.

<existing_code>
{existing_code}
</existing_code>

When generating, preserve functional parts that still apply,
and refactor only what the new brief requires.
"""

    base += "\nDELIVERABLE: provide only code blocks; no explanations."
    return base


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
def generate_app_with_gemini(brief: str, work_dir: str, attachment_names: List[str]) -> None:
    """
    Calls Gemini with the brief and writes the returned files into work_dir.
    At minimum writes index.html; may also write script.js/style.css if present.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(DEFAULT_MODEL)

    # 🧩 Added for Round-2 awareness
    # Read any existing files in the work_dir (in round-2, repo is cloned here)
    existing_files = []
    for p in pathlib.Path(work_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in (".html", ".css", ".js"):
            try:
                content = p.read_text(encoding="utf-8")
                existing_files.append(f"\n--- {p.name} ---\n{content}")
            except Exception:
                pass
    existing_code = "\n".join(existing_files)

    prompt = _prompt_for_webapp(brief, attachment_names, existing_code)
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=4096,
        ),
    )

    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty response")

    files = _extract_files_from_markdown(text)
    if "index.html" not in files:
        files["index.html"] = next(iter(files.values()))

    root = pathlib.Path(work_dir)
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
