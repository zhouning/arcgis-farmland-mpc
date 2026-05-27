"""FastAPI single-page form UI for the farmland-mpc pipeline.

Workflow:
  1. POST /run with a multipart upload (DLTB shp + companions, optional XZQ)
     and a chosen UTM CRS. We assign a job_id (uuid4), kick off the four CLI
     phases in a background task, and return the job_id.
  2. GET /status/{job_id} reports phase, % progress, and tail-lines of the log.
  3. GET /download/{job_id} streams optimized.shp + companions as a single zip
     once the run is done.
  4. GET / renders the form + a live-polling status panel (template).

Designed for the Docker `webui` mode — single-tenant, single-host. No auth.
For multi-user / internet-exposed deployments, put a reverse proxy with auth
in front (the docker-compose example shows nginx + basic-auth).
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


WORK = Path("/work").resolve()
JOBS_ROOT = WORK / "_webui_jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

PKG_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(PKG_DIR / "templates"))

logger = logging.getLogger("farmland_mpc.webui")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="farmland-mpc", version="0.1")
app.mount("/static", StaticFiles(directory=str(PKG_DIR / "static")), name="static")


@dataclass
class Job:
    id: str
    crs: str
    horizon: int
    top_k: int
    n_episodes: int
    phase: str = "queued"
    progress: float = 0.0
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    error: Optional[str] = None

    @property
    def root(self) -> Path:
        return JOBS_ROOT / self.id

    @property
    def in_dir(self) -> Path:
        return self.root / "in"

    @property
    def prepared_dir(self) -> Path:
        return self.root / "prepared"

    @property
    def out_dir(self) -> Path:
        return self.root / "mpc_output"

    @property
    def log_path(self) -> Path:
        return self.root / "run.log"


JOBS: dict[str, Job] = {}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"jobs": JOBS})


@app.post("/run")
async def run(
    background_tasks: BackgroundTasks,
    crs: str = Form("EPSG:32648"),
    horizon: int = Form(5),
    top_k: int = Form(50),
    n_episodes: int = Form(5),
    dltb: list[UploadFile] = File(...),
):
    job = Job(id=str(uuid.uuid4())[:8], crs=crs, horizon=horizon,
              top_k=top_k, n_episodes=n_episodes)
    job.in_dir.mkdir(parents=True, exist_ok=True)
    job.out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for f in dltb:
        # f.filename is user-supplied; sanitize aggressively.
        safe = Path(f.filename or "uploaded").name
        if not safe or safe.startswith("."):
            raise HTTPException(400, f"invalid filename: {f.filename!r}")
        dst = job.in_dir / safe
        with open(dst, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(dst)

    shps = [p for p in saved_paths if p.suffix.lower() == ".shp"]
    if len(shps) != 1:
        raise HTTPException(400, f"expected exactly one .shp upload, got {len(shps)}")
    job_dltb = shps[0]

    JOBS[job.id] = job
    background_tasks.add_task(_run_pipeline, job, job_dltb)
    return {"job_id": job.id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    log_tail = ""
    if job.log_path.exists():
        log_tail = "\n".join(job.log_path.read_text(errors="replace").splitlines()[-30:])
    return {
        "id": job.id, "phase": job.phase, "progress": job.progress,
        "error": job.error,
        "elapsed_s": (job.finished or time.time()) - job.started,
        "log_tail": log_tail,
    }


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.phase != "done":
        raise HTTPException(409, f"job not finished (phase={job.phase})")

    zip_path = job.root / "optimized.zip"
    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in job.out_dir.glob("optimized.*"):
                zf.write(p, p.name)
            summary = job.out_dir / "mpc_summary.json"
            if summary.exists():
                zf.write(summary, summary.name)
    return FileResponse(zip_path, filename=f"farmland_mpc_{job.id}.zip",
                        media_type="application/zip")


# --------------------------------------------------------------------------- #
# Pipeline (runs in background task)
# --------------------------------------------------------------------------- #

def _run_pipeline(job: Job, dltb: Path) -> None:
    try:
        with open(job.log_path, "ab", buffering=0) as logf:
            def shell(cmd: list[str], phase: str, progress_after: float):
                job.phase = phase
                logf.write(f"\n=== {phase} ===\n$ {' '.join(cmd)}\n".encode())
                logf.flush()
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                      check=False)
                if proc.returncode != 0:
                    raise RuntimeError(f"{phase} failed (exit {proc.returncode})")
                job.progress = progress_after

            # Phase 0 — fetch DEM
            dem_tif = job.root / "dem.tif"
            shell([sys.executable, "/repo/scripts/fetch_dem.py",
                   "--dltb", str(dltb),
                   "--work-dir", str(job.root),
                   "--proj-crs", job.crs],
                  "fetch_dem", 0.10)

            # Phase A — prepare
            shell(["farmland-mpc", "prepare",
                   "--dltb", str(dltb),
                   "--dem", str(dem_tif),
                   "--out", str(job.prepared_dir),
                   "--crs", job.crs],
                  "prepare", 0.20)

            # Phase B — sample
            shell(["farmland-mpc", "sample",
                   "--prepared-dir", str(job.prepared_dir),
                   "--n-episodes", "60",
                   "--n-states", "1000",
                   "--n-actions", "50",
                   "--seed", "0"],
                  "sample", 0.45)

            # Phase C — train
            shell(["farmland-mpc", "train",
                   "--prepared-dir", str(job.prepared_dir),
                   "--epochs", "30",
                   "--lambda-rank", "5.0",
                   "--n-members", "3"],
                  "train", 0.80)

            # Phase D — plan
            shell(["farmland-mpc", "plan",
                   "--prepared-dir", str(job.prepared_dir),
                   "--ensemble-dir", str(job.prepared_dir / "tool3"),
                   "--out-dir", str(job.out_dir),
                   "--output-shp", str(job.out_dir / "optimized.shp"),
                   "--crs", job.crs,
                   "--horizon", str(job.horizon),
                   "--top-k", str(job.top_k),
                   "--continuation", "greedy",
                   "--n-episodes", str(job.n_episodes)],
                  "plan", 1.00)

        job.phase = "done"
        job.finished = time.time()
    except Exception as e:
        job.phase = "error"
        job.error = str(e)
        job.finished = time.time()
        logger.exception("job %s failed", job.id)
