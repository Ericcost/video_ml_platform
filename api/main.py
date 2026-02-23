"""
main.py — Volleyball Analyzer REST API
Endpoints:
  POST /upload          Upload a video and queue analysis job
  GET  /status/{job_id} Poll job status + progress
  GET  /result/{job_id} Get full analysis result
  GET  /video/{job_id}  Stream annotated output video
  GET  /health          Health check
"""

import io
import os
import uuid
import logging
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from celery import Celery
from pymongo import MongoClient
from minio import Minio
from minio.error import S3Error

from models import JobResponse, JobStatus, AnalysisResult

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL",   "redis://localhost:6379/0")
MONGO_URL   = os.getenv("MONGO_URL",   "mongodb://localhost:27017")
MINIO_URL   = os.getenv("MINIO_URL",   "localhost:9000")
MINIO_KEY   = os.getenv("MINIO_ACCESS_KEY", "volleyball")
MINIO_SEC   = os.getenv("MINIO_SECRET_KEY", "volleyball123")

BUCKET_RAW  = "raw-videos"
BUCKET_OUT  = "output-videos"
ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_SIZE_MB = 500

# ── Clients ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Volleyball Analyzer API",
    description="Upload volleyball videos and get action analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

celery_app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_routes = {"worker.tasks.*": {"queue": "default"}}

mongo   = MongoClient(MONGO_URL)
db      = mongo["volleyball"]
jobs_col = db["jobs"]

minio_client = Minio(
    MINIO_URL,
    access_key=MINIO_KEY,
    secret_key=MINIO_SEC,
    secure=False,
)


def ensure_buckets():
    for bucket in [BUCKET_RAW, BUCKET_OUT]:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)


@app.on_event("startup")
async def startup():
    try:
        ensure_buckets()
        log.info("MinIO buckets ready.")
    except Exception as e:
        log.warning(f"MinIO not ready yet: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/upload", response_model=JobResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file, store it in MinIO, and queue an analysis job.
    Returns a job_id to poll for results.
    """
    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format: {ext}. Use {ALLOWED_EXT}")

    # Read + size check
    content = await file.read()
    size_mb = len(content) / (1024 ** 2)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max {MAX_SIZE_MB} MB.")

    job_id   = str(uuid.uuid4())
    obj_name = f"{job_id}{ext}"

    # Upload to MinIO
    try:
        ensure_buckets()
        minio_client.put_object(
            BUCKET_RAW, obj_name,
            io.BytesIO(content), len(content),
            content_type=file.content_type or "video/mp4",
        )
    except S3Error as e:
        log.error(f"MinIO upload error: {e}")
        raise HTTPException(500, "Storage error, please retry.")

    # Persist job in MongoDB
    jobs_col.insert_one({
        "_id":       job_id,
        "status":    JobStatus.PENDING,
        "filename":  file.filename,
        "object":    obj_name,
        "progress":  0.0,
        "created_at": datetime.utcnow(),
    })

    # Queue Celery task
    celery_app.send_task(
        "worker.tasks.analyze_video",
        args=[job_id, obj_name],
        queue="default",
    )

    log.info(f"Job {job_id} queued for {file.filename}")

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Video uploaded. Analysis queued.",
        progress=0.0,
    )


@app.get("/status/{job_id}", response_model=JobResponse)
def get_status(job_id: str):
    """Poll job status and progress (0.0 → 1.0)."""
    job = jobs_col.find_one({"_id": job_id})
    if not job:
        raise HTTPException(404, f"Job {job_id} not found.")

    result = None
    if job["status"] == JobStatus.DONE and job.get("result"):
        result = AnalysisResult(**job["result"])

    return JobResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("message", ""),
        progress=job.get("progress", 0.0),
        result=result,
    )


@app.get("/result/{job_id}", response_model=AnalysisResult)
def get_result(job_id: str):
    """Get the full analysis result (only available when status == done)."""
    job = jobs_col.find_one({"_id": job_id})
    if not job:
        raise HTTPException(404, f"Job {job_id} not found.")
    if job["status"] != JobStatus.DONE:
        raise HTTPException(409, f"Job not finished yet. Status: {job['status']}")
    if not job.get("result"):
        raise HTTPException(500, "Job finished but result missing.")
    return AnalysisResult(**job["result"])


@app.get("/video/{job_id}")
def stream_video(job_id: str):
    """Stream the annotated output video from MinIO."""
    job = jobs_col.find_one({"_id": job_id})
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] != JobStatus.DONE:
        raise HTTPException(409, "Video not ready yet.")

    obj_name = job.get("output_object")
    if not obj_name:
        raise HTTPException(404, "Output video not found.")

    try:
        response = minio_client.get_object(BUCKET_OUT, obj_name)
        return StreamingResponse(
            response,
            media_type="video/mp4",
            headers={"Content-Disposition": f'inline; filename="analyzed_{job_id}.mp4"'},
        )
    except S3Error as e:
        raise HTTPException(500, f"Could not retrieve video: {e}")
