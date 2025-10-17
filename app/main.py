from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Tuple
import os, time, base64, subprocess, tempfile, pathlib, requests

app = FastAPI(title="Student API – LLM Code Deployment")

# ------------------ Models ------------------
class Attachment(BaseModel):
    name: str
    url: str  # data:... or https://...

class Task(BaseModel):
    email: str
    secret: str
    task: str
    round: int
    nonce: str
    brief: str
    checks: List[str]
    evaluation_url: HttpUrl
    attachments: Optional[List[Attachment]] = []

# ------------------ Config ------------------
SHARED_SECRET   = os.getenv("STUDENT_SHARED_SECRET")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")
GITHUB_USER     = os.getenv("GITHUB_USER")
DEFAULT_BRANCH  = os.getenv("DEFAULT_BRANCH", "main")

if not SHARED_SECRET:
    print("⚠️ STUDENT_SHARED_SECRET not set")
if not GITHUB_TOKEN or not GITHUB_USER:
    print("⚠️ GITHUB_TOKEN or GITHUB_USER not set (pushes will fail)")

# ------------------ Helpers ------------------
def verify_secret(s: str):
    if not SHARED_SECRET or s != SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid secret")

def decode_data_uri(data_uri: str) -> bytes:
    if data_uri.startswith("data:"):
        b64 = data_uri.split(",", 1)[1]
        return base64.b64decode(b64)
    return requests.get(data_uri, timeout=30).content

def run(cmd: list[str], cwd: Optional[str] = None):
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"CMD failed: {' '.join(cmd)}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return res.stdout.strip()

def wait_for_pages(url: str, max_wait: int = 240, interval: int = 8) -> bool:
    """Poll the Pages URL until it returns 200 or timeout."""
    print(f"⏳ Waiting for GitHub Pages: {url}")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                print("✅ GitHub Pages is live")
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
    print("⚠️ Timed out waiting for GitHub Pages")
    return False

# ------------------ GitHub + Pages ------------------
def _gh_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def _token_remote(repo: str) -> str:
    # tokenized remote for CI push
    return f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{repo}.git"

def _public_remote(repo: str) -> str:
    # non-token remote to avoid leaving token in config
    return f"https://github.com/{GITHUB_USER}/{repo}.git"

def create_repo_and_push(task_id: str, app_dir: str) -> Tuple[str, str, str]:
    repo = f"{task_id}"
    repo_url = _public_remote(repo)

    # Init repo on the correct branch
    run(["git", "init", "-b", DEFAULT_BRANCH], cwd=app_dir)
    run(["git", "config", "user.email", "bot@local"], cwd=app_dir)
    run(["git", "config", "user.name", "Bot"], cwd=app_dir)

    # Minimal LICENSE + README
    (pathlib.Path(app_dir) / "LICENSE").write_text(
        "MIT License\n\nGenerated automatically.", encoding="utf-8"
    )
    readme = pathlib.Path(app_dir) / "README.md"
    if not readme.exists():
        readme.write_text("# Auto-generated App\n\nSee LICENSE.", encoding="utf-8")

    # GitHub Pages workflow on DEFAULT_BRANCH
    wf_dir = pathlib.Path(app_dir) / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "pages.yml").write_text(PAGES_WF_YML, encoding="utf-8")

    run(["git", "add", "-A"], cwd=app_dir)
    run(["git", "commit", "-m", "init with Pages workflow"], cwd=app_dir)

    # Create repo via REST (idempotent)
    r = requests.post("https://api.github.com/user/repos", headers=_gh_headers(),
                      json={"name": repo, "private": False, "auto_init": False})
    if r.status_code == 422 and "already exists" in r.text.lower():
        print(f"ℹ️ Repo {repo} already exists, continuing.")
    elif r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub repo creation failed: {r.status_code} {r.text}")

    # Push using tokenized remote (Render has no interactive creds)
    run(["git", "remote", "add", "origin", _token_remote(repo)], cwd=app_dir)
    run(["git", "push", "-u", "origin", DEFAULT_BRANCH], cwd=app_dir)
    # Clean remote URL (avoid token lingering in config)
    run(["git", "remote", "set-url", "origin", repo_url], cwd=app_dir)

    # Enable Pages via REST API (workflow build)
    for _ in range(5):
        try:
            resp = requests.post(
                f"https://api.github.com/repos/{GITHUB_USER}/{repo}/pages",
                headers=_gh_headers(),
                json={"build_type": "workflow"},
                timeout=15,
            )
            if resp.status_code in (201, 204, 409):
                break
        except Exception:
            pass
        time.sleep(2)

    sha = run(["git", "rev-parse", "HEAD"], cwd=app_dir)
    pages_url = f"https://{GITHUB_USER}.github.io/{repo}/"
    return repo_url, sha, pages_url

PAGES_WF_YML = f"""name: Deploy to Pages
on:
  push:
    branches: [ {os.getenv('DEFAULT_BRANCH','main')} ]
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: mkdir -p dist && cp -r . dist/ || true
      - uses: actions/upload-pages-artifact@v3
        with:
          path: dist
      - uses: actions/deploy-pages@v4
"""

# ------------------ Gemini ------------------
from .llm_generator_gemini import generate_app_with_gemini

def generate_llm_app(brief: str, attachments, task_id: str, round_idx: int) -> str:
    tmp = tempfile.mkdtemp(prefix=f"app-{task_id}-r{round_idx}-")
    root = pathlib.Path(tmp)
    names = []
    for att in attachments or []:
        (root / att.name).write_bytes(decode_data_uri(att.url))
        names.append(att.name)

    # Call Gemini (will fallback to a minimal page if rate-limited)
    generate_app_with_gemini(brief, tmp, names)

    # Ensure README
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(f"# Auto App for {task_id}\n\nBrief: {brief}\n", encoding="utf-8")
    return tmp

def update_llm_app(task: Task) -> Tuple[str, str]:
    """Clone, update with Gemini, commit & push to DEFAULT_BRANCH. Returns (tmp_dir, commit_sha)."""
    repo = f"{task.task}"
    tmp = tempfile.mkdtemp(prefix=f"update-{repo}-")

    # Clone
    run(["git", "clone", _public_remote(repo), tmp])
    # Switch to branch (track remote if needed)
    try:
        run(["git", "checkout", DEFAULT_BRANCH], cwd=tmp)
    except RuntimeError:
        run(["git", "checkout", "-B", DEFAULT_BRANCH, f"origin/{DEFAULT_BRANCH}"], cwd=tmp)

    # Update with Gemini (fallback-safe)
    generate_app_with_gemini(task.brief, tmp, [a.name for a in task.attachments or []])

    # Ensure workflow exists (in case repo was missing it)
    wf_dir = pathlib.Path(tmp) / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "pages.yml").write_text(PAGES_WF_YML, encoding="utf-8")

    # Guarantee a diff to retrigger Actions
    (pathlib.Path(tmp) / ".redeploy").write_text(str(time.time()), encoding="utf-8")

    # README touch
    (pathlib.Path(tmp) / "README.md").write_text(f"""# Updated Auto App – {task.task}

**Round:** {task.round}  
**Brief:** {task.brief}

Automated update & redeploy via Gemini.
""", encoding="utf-8")

    # Tokenize remote for push, push, then clean
    run(["git", "remote", "set-url", "origin", _token_remote(repo)], cwd=tmp)

    # ⚙️ Set commit identity (Render has no global git config)
    run(["git", "config", "user.email", "bot@local"], cwd=tmp)
    run(["git", "config", "user.name", "Bot"], cwd=tmp)

    run(["git", "add", "-A"], cwd=tmp)
    run(["git", "commit", "-m", f"round {task.round}: update app"], cwd=tmp)
    run(["git", "push", "origin", DEFAULT_BRANCH], cwd=tmp)
    run(["git", "remote", "set-url", "origin", _public_remote(repo)], cwd=tmp)

    sha = run(["git", "rev-parse", "HEAD"], cwd=tmp)
    return tmp, sha


# ------------------ Evaluator notify ------------------
def notify_evaluator(evaluation_url: str, payload: dict):
    # Optional: wait for Pages to be live before notifying
    wait_for_pages(payload["pages_url"])
    for delay in [1, 2, 4, 8, 16]:
        try:
            r = requests.post(evaluation_url, json=payload,
                              headers={"Content-Type": "application/json"},
                              timeout=30)
            if r.status_code == 200:
                print("✅ Evaluator notified successfully.")
                return
            print(f"⚠️ Evaluator returned {r.status_code}, retrying...")
        except Exception as e:
            print(f"⚠️ Notify failed ({e}), retrying...")
        time.sleep(delay)
    print("❌ Failed to notify evaluator after retries.")

# ------------------ Endpoints ------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "llm-code-deployment",
        "ingest": "/ingest",
        "default_branch": DEFAULT_BRANCH,
    }

@app.post("/ingest")
async def ingest(task: Task, req: Request, background_tasks: BackgroundTasks):
    verify_secret(task.secret)

    if task.round == 1:
        app_dir = generate_llm_app(task.brief, task.attachments or [], task.task, task.round)
        repo_url, sha, pages_url = create_repo_and_push(task.task, app_dir)
    else:
        app_dir, sha = update_llm_app(task)
        repo_url  = _public_remote(task.task)
        pages_url = f"https://{GITHUB_USER}.github.io/{task.task}/"

    payload = {
        "email": task.email,
        "task": task.task,
        "round": task.round,
        "nonce": task.nonce,
        "repo_url": repo_url,
        "commit_sha": sha,
        "pages_url": pages_url
    }

    # Notify in background; return HTTP 200 immediately
    background_tasks.add_task(notify_evaluator, str(task.evaluation_url), payload)

    return {"ok": True, "repo_url": repo_url, "commit_sha": sha, "pages_url": pages_url}
