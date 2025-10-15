from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import os, json, time, base64, subprocess, tempfile, pathlib, requests

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
DEFAULT_BRANCH  = os.getenv("DEFAULT_BRANCH", "main")  # <<<< one branch everywhere

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
        raise RuntimeError(f"CMD failed: {' '.join(cmd)}\n{res.stdout}\n{res.stderr}")
    return res.stdout.strip()

# ------------------ GitHub + Pages ------------------
def create_repo_and_push(task_id: str, app_dir: str) -> tuple[str, str, str]:
    repo = f"{task_id}"
    repo_url = f"https://github.com/{GITHUB_USER}/{repo}"

    # init repo ON THE RIGHT BRANCH
    run(["git", "init", "-b", DEFAULT_BRANCH], cwd=app_dir)
    run(["git", "config", "user.email", "bot@local"], cwd=app_dir)
    run(["git", "config", "user.name", "Bot"], cwd=app_dir)

    # LICENSE
    mit = pathlib.Path(app_dir) / "LICENSE"
    mit_text_path = (pathlib.Path(__file__).parent / ".." / "shared" / "mit.txt").resolve()
    if mit_text_path.exists():
        mit.write_text("MIT License\n\n" + mit_text_path.read_text(encoding="utf-8"))
    else:
        mit.write_text("MIT License\n\nGenerated automatically.", encoding="utf-8")

    # README
    readme = pathlib.Path(app_dir) / "README.md"
    if not readme.exists():
        readme.write_text("# Auto-generated App\n\nSee LICENSE.", encoding="utf-8")

    # add workflow BEFORE first push so default branch has it
    wf_dir = pathlib.Path(app_dir) / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "pages.yml").write_text(PAGES_WF_YML, encoding="utf-8")

    # first commit
    run(["git", "add", "-A"], cwd=app_dir)
    run(["git", "commit", "-m", "init with Pages workflow"], cwd=app_dir)

    # create GH repo from current dir; default branch will be our HEAD (DEFAULT_BRANCH)
    run([
        "gh", "repo", "create", f"{GITHUB_USER}/{repo}",
        "--public", "--source=.", "--remote=origin", "--push", "--confirm"
    ], cwd=app_dir)

    # enable pages for workflow builds (retry a bit)
    for i in range(5):
        try:
            run(["gh", "api", f"repos/{GITHUB_USER}/{repo}/pages",
                 "-X", "POST", "-f", "build_type=workflow"], cwd=app_dir)
            break
        except Exception:
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
    generate_app_with_gemini(brief, tmp, names)

    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(f"# Auto App for {task_id}\n\nBrief: {brief}\n", encoding="utf-8")
    return tmp

def update_llm_app(task: Task) -> str:
    repo = f"{task.task}"
    tmp = tempfile.mkdtemp(prefix=f"update-{repo}-")

    # Clone and switch to the ONE branch we use
    run(["git", "clone", f"https://github.com/{GITHUB_USER}/{repo}.git", tmp])
    run(["git", "checkout", DEFAULT_BRANCH], cwd=tmp)

    # Re-generate files with Gemini using the new brief
    generate_app_with_gemini(task.brief, tmp, [a.name for a in task.attachments or []])

    # Touch a file to guarantee a diff (ensures Actions re-runs)
    (pathlib.Path(tmp) / ".redeploy").write_text(str(time.time()), encoding="utf-8")

    # README update
    (pathlib.Path(tmp) / "README.md").write_text(f"""# Updated Auto App – {task.task}

**Round:** {task.round}  
**Brief:** {task.brief}

Automated update & redeploy via Gemini.

## License
MIT License
""", encoding="utf-8")

    # Commit + push to the SAME branch
    run(["git", "add", "-A"], cwd=tmp)
    run(["git", "commit", "-m", f"round {task.round}: update app"], cwd=tmp)
    run(["git", "push", "origin", DEFAULT_BRANCH], cwd=tmp)
    return tmp

# ------------------ Evaluator notify ------------------
def notify_evaluator(evaluation_url: str, payload: dict):
    for delay in [1, 2, 4, 8, 16]:
        try:
            r = requests.post(evaluation_url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            if r.status_code == 200:
                print("✅ Evaluator notified successfully.")
                return
            print(f"⚠️ Evaluator returned {r.status_code}, retrying...")
        except Exception as e:
            print(f"⚠️ Notify failed ({e}), retrying...")
        time.sleep(delay)
    raise RuntimeError("Failed to notify evaluator after retries.")

# ------------------ Endpoint ------------------
@app.post("/ingest")
async def ingest(task: Task, req: Request):
    verify_secret(task.secret)

    if task.round == 1:
        app_dir = generate_llm_app(task.brief, task.attachments or [], task.task, task.round)
        repo_url, sha, pages_url = create_repo_and_push(task.task, app_dir)
    else:
        app_dir = update_llm_app(task)
        repo_url = f"https://github.com/{GITHUB_USER}/{task.task}"
        sha = run(["git", "rev-parse", "HEAD"], cwd=app_dir)
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
    notify_evaluator(str(task.evaluation_url), payload)
    return {"ok": True, "repo_url": repo_url, "commit_sha": sha, "pages_url": pages_url}
