"""
tasks.py
--------
Celery tasks for the Volleyball Analyzer worker.
"""

import io
import os
import logging
import tempfile
from datetime import datetime

from celery import Celery
from pymongo import MongoClient
from minio import Minio

from video_processor import process_video

log = logging.getLogger(__name__)

REDIS_URL  = os.getenv("REDIS_URL",  "redis://localhost:6379/0")
MONGO_URL  = os.getenv("MONGO_URL",  "mongodb://localhost:27017")
MINIO_URL  = os.getenv("MINIO_URL",  "localhost:9000")
MINIO_KEY  = os.getenv("MINIO_ACCESS_KEY", "volleyball")
MINIO_SEC  = os.getenv("MINIO_SECRET_KEY", "volleyball123")

BUCKET_RAW = "raw-videos"
BUCKET_OUT = "output-videos"

app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL)

mongo        = MongoClient(MONGO_URL)
db           = mongo["volleyball"]
jobs_col     = db["jobs"]

minio_client = Minio(MINIO_URL, access_key=MINIO_KEY, secret_key=MINIO_SEC, secure=False)


def _update_job(job_id: str, **kwargs):
    jobs_col.update_one({"_id": job_id}, {"$set": kwargs})


@app.task(name="worker.tasks.analyze_video", bind=True, max_retries=2)
def analyze_video(self, job_id: str, object_name: str):
    """
    Main Celery task:
      1. Download raw video from MinIO
      2. Run detection + tracking + classification pipeline
      3. Upload annotated video to MinIO
      4. Store structured result in MongoDB
    """
    log.info(f"[{job_id}] Starting analysis of {object_name}")

    _update_job(job_id, status="processing", progress=0.0,
                message="Downloading video...", started_at=datetime.utcnow())

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "output.mp4")

        # ── Step 1: Download raw video ────────────────────────────────
        try:
            minio_client.fget_object(BUCKET_RAW, object_name, input_path)
            log.info(f"[{job_id}] Video downloaded: {os.path.getsize(input_path)} bytes")
        except Exception as e:
            log.error(f"[{job_id}] Download failed: {e}")
            _update_job(job_id, status="error", message=f"Download failed: {e}")
            raise

        _update_job(job_id, progress=0.05, message="Running AI analysis...")

        # ── Step 2: Run analysis pipeline ────────────────────────────
        def progress_cb(p: float):
            _update_job(job_id, progress=0.05 + p * 0.85,
                        message=f"Analysing... {int(p*100)}%")

        try:
            result = process_video(
                input_path=input_path,
                output_path=output_path,
                progress_callback=progress_cb,
                frame_skip=2,
            )
        except Exception as e:
            log.error(f"[{job_id}] Processing failed: {e}")
            _update_job(job_id, status="error", message=f"Processing failed: {e}")
            raise

        # ── Step 3: Upload annotated video ────────────────────────────
        _update_job(job_id, progress=0.92, message="Uploading annotated video...")

        output_object = f"{job_id}_annotated.mp4"
        try:
            minio_client.fput_object(BUCKET_OUT, output_object, output_path,
                                     content_type="video/mp4")
        except Exception as e:
            log.warning(f"[{job_id}] Upload of annotated video failed: {e}")
            output_object = None

        # ── Step 4: Build & store result ──────────────────────────────
        job = jobs_col.find_one({"_id": job_id})
        full_result = {
            "job_id":           job_id,
            "video_filename":   job.get("filename", object_name),
            "duration":         result["duration"],
            "fps":              result["fps"],
            "total_frames":     result["total_frames"],
            "processed_frames": result["processed_frames"],
            "events":           result["events"],
            "team_a_color":     result["team_a_color"],
            "team_b_color":     result["team_b_color"],
            "output_video_url": f"/video/{job_id}" if output_object else None,
        }

        _update_job(
            job_id,
            status="done",
            progress=1.0,
            message=f"Done. {len(result['events'])} events detected.",
            result=full_result,
            output_object=output_object,
            completed_at=datetime.utcnow(),
        )

        log.info(f"[{job_id}] Analysis complete. {len(result['events'])} events.")
        return job_id
