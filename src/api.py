import sys
import os
import time
import shutil
import asyncio
import traceback
import numpy as np
import cv2
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid

from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST, push_to_gateway,
    REGISTRY
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Make src/ importable regardless of cwd
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import load_config, get_project_root
from async_logger import AsyncBatchLogger
from drift_handler import save_feedback_features, run_finetuning

config = load_config()
IMG_SIZE   = config['training']['img_size']
CLASSES    = config['training']['classes']
MODEL_PATH = os.path.join(get_project_root(), "models", "fire_model.keras")

log_config = config.get("logging", {})
api_logger = AsyncBatchLogger(
    batch_size=log_config.get("batch_size", 20),
    time_limit_sec=log_config.get("time_limit_sec", 5.0),
    log_file=os.path.join(get_project_root(), log_config.get("log_file", "logs/api_requests.jsonl"))
)

# ─────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Model inference latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)
INFERENCE_COUNT = Counter("inference_total", "Total inference calls")
TRAINING_STATUS = Gauge("training_in_progress", "Whether training is currently running")
RETRAIN_COUNT   = Counter("retrain_total", "Total retrain calls")

PUSHGATEWAY_URL = config.get("monitoring", {}).get("pushgateway_url", "localhost:9091")

def push_metrics():
    """Push current metrics to Prometheus Pushgateway."""
    try:
        push_to_gateway(PUSHGATEWAY_URL, job="edgeguard_api", registry=REGISTRY)
    except Exception as e:
        print(f"⚠️  Could not push to Prometheus: {e}")


# ─────────────────────────────────────────────
# Prometheus Middleware
# ─────────────────────────────────────────────
class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        endpoint = request.url.path
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code
        ).inc()

        # Log request asynchronously
        await api_logger.log({
            "method": request.method,
            "endpoint": endpoint,
            "status_code": response.status_code,
            "latency_ms": round(duration * 1000, 3)
        })

        return response


# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
model_store: dict = {"model": None}
pipeline_state: dict = {"running": False, "status": "idle", "message": ""}
executor = ThreadPoolExecutor(max_workers=2)

pending_feedback = {}
FEEDBACK_TTL = 300 # 5 minutes



def load_model_into_store():
    if os.path.exists(MODEL_PATH):
        model_store["model"] = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  No trained model found at {MODEL_PATH}. Train one first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_into_store()
    api_logger.start()
    yield
    await api_logger.stop()


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="EdgeGuard-AI Inference API",
    description=(
        "REST API for EdgeGuard-AI fire-detection pipeline. "
        "Supports training, inference and retraining."
    ),
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(PrometheusMiddleware)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid JPEG/PNG file.")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img.astype(np.float32)


def _run_pipeline_sync():
    """Blocking pipeline execution (runs in thread)."""
    from pipeline import run_pipeline   # imported here to avoid circular deps at module level
    run_pipeline()


def _run_retrain_sync(dataset_dir: str):
    """Blocking retrain execution on a custom dataset."""
    # Temporarily override the dataset_path in the config for this run
    import yaml
    config_path = os.path.join(get_project_root(), "config", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    original_path = cfg["dataset_path"]
    cfg["dataset_path"] = dataset_dir
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    try:
        _run_pipeline_sync()
    finally:
        # Restore original path
        cfg["dataset_path"] = original_path
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)
        # Reload model after retraining
        load_model_into_store()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "EdgeGuard-AI Inference API",
        "version": "1.0.0",
        "endpoints": {
            "GET  /":               "This help message",
            "GET  /health":         "Service health check",
            "GET  /metrics":        "Prometheus metrics",
            "POST /train":          "Start the training pipeline",
            "POST /infer":          "Run inference on an uploaded image",
            "POST /retrain":        "Retrain on a new dataset folder",
            "GET  /pipeline/status":"Check training / retraining status"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    model_ready = model_store["model"] is not None
    return {
        "status": "ok",
        "model_loaded": model_ready,
        "pipeline_running": pipeline_state["running"]
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    push_metrics()
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ── 1. Train ──────────────────────────────────
@app.post("/train", tags=["Pipeline"])
async def start_training(background_tasks: BackgroundTasks):
    """
    Starts the full training pipeline in the background.

    - Loads the dataset configured in `config/config.yaml` (`fire_dataset/`).
    - Trains the edge-optimised fire-detection model.
    - Logs all metrics to MLflow (http://localhost:5000).
    - Saves `models/fire_model.h5` and `models/fire_model_int8.tflite`.
    - Pushes training metrics to Prometheus.
    """
    if pipeline_state["running"]:
        raise HTTPException(status_code=409, detail="A pipeline run is already in progress.")

    def _task():
        pipeline_state["running"] = True
        pipeline_state["status"]  = "training"
        pipeline_state["message"] = "Training started …"
        TRAINING_STATUS.set(1)
        try:
            _run_pipeline_sync()
            load_model_into_store()
            pipeline_state["status"]  = "completed"
            pipeline_state["message"] = "Training finished successfully."
        except Exception as exc:
            pipeline_state["status"]  = "failed"
            pipeline_state["message"] = str(exc)
            traceback.print_exc()
        finally:
            pipeline_state["running"] = False
            TRAINING_STATUS.set(0)
            push_metrics()

    background_tasks.add_task(_task)

    return {
        "message": "Training pipeline started in the background.",
        "monitor_at": "http://localhost:5000",
        "status_endpoint": "/pipeline/status",
        "note": (
            "Training may take several minutes depending on dataset size. "
            "Poll /pipeline/status for live updates."
        )
    }


# ── 2. Infer ─────────────────────────────────
@app.post("/infer", tags=["Inference"])
async def infer(file: UploadFile = File(...)):
    """
    Runs fire-detection inference on a single image.

    **Image requirements**
    - Format  : JPEG or PNG
    - Channels: RGB (colour)
    - Size    : any – will be auto-resized to 128 × 128
    - Content : should be an outdoor / scene photo (not a document scan)

    Returns the predicted class (`fire` / `no_fire`) and confidence score.
    """
    if model_store["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model found. Call POST /train first."
        )

    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG image."
        )

    image_bytes = await file.read()

    try:
        img = preprocess_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    img_batch = np.expand_dims(img, 0)

    t0 = time.time()
    preds = model_store["model"].predict(img_batch, verbose=0)
    latency = time.time() - t0

    INFERENCE_LATENCY.observe(latency)
    INFERENCE_COUNT.inc()
    push_metrics()

    class_idx   = int(np.argmax(preds[0]))
    confidence  = float(preds[0][class_idx])
    class_label = CLASSES[class_idx]

    response_data = {
        "prediction":   class_label,
        "confidence":   round(confidence * 100, 2),
        "all_scores":   {c: round(float(s) * 100, 2) for c, s in zip(CLASSES, preds[0])},
        "latency_ms":   round(latency * 1000, 3),
        "image_resized_to": f"{IMG_SIZE}x{IMG_SIZE}",
    }

    if confidence < 0.80:
        feedback_id = str(uuid.uuid4())
        # Clean up old entries
        now = time.time()
        keys_to_delete = [k for k, v in pending_feedback.items() if now - v[0] > FEEDBACK_TTL]
        for k in keys_to_delete:
            del pending_feedback[k]
            
        pending_feedback[feedback_id] = (now, img)
        response_data["needs_feedback"] = True
        response_data["feedback_id"] = feedback_id

    return response_data


# ── 3. Retrain ────────────────────────────────
@app.post("/retrain", tags=["Pipeline"])
async def retrain(background_tasks: BackgroundTasks, dataset_dir: str = None):
    """
    Retrains the model on a **new dataset** placed on the server.

    ---
    ### 📁 Dataset Format Requirements

    Your dataset folder **must** follow this exact structure:

    ```
    <your_dataset_folder>/
    ├── fire_images/       ← images containing FIRE
    │   ├── img001.jpg
    │   ├── img002.png
    │   └── ...
    └── non_fire_images/   ← images WITHOUT fire
        ├── img001.jpg
        ├── img002.png
        └── ...
    ```

    ### 🖼️ Image Requirements

    | Property | Requirement |
    |----------|-------------|
    | Format   | JPEG (.jpg / .jpeg) or PNG (.png) |
    | Channels | RGB colour (3 channels) |
    | Min size | 64 × 64 pixels |
    | Max size | No hard limit (auto-resized to 128×128) |
    | Min images per class | 50 (more = better accuracy) |

    ### 🚀 Usage

    Pass the **absolute path** to your dataset folder as the `dataset_dir` query parameter:

    ```
    POST /retrain?dataset_dir=D:/my_new_dataset
    ```

    Leave `dataset_dir` empty to retrain on the default `fire_dataset/`.
    """
    if pipeline_state["running"]:
        raise HTTPException(status_code=409, detail="A pipeline run is already in progress.")

    root = get_project_root()
    target_dir = dataset_dir if dataset_dir else os.path.join(root, config["dataset_path"])

    # Validate structure
    fire_dir     = os.path.join(target_dir, "fire_images")
    no_fire_dir  = os.path.join(target_dir, "non_fire_images")
    errors = []
    if not os.path.isdir(fire_dir):
        errors.append(f"Missing folder: {fire_dir}")
    if not os.path.isdir(no_fire_dir):
        errors.append(f"Missing folder: {no_fire_dir}")
    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid dataset structure.",
                "issues": errors,
                "required_structure": {
                    f"{target_dir}/fire_images/":     "Images WITH fire (jpg/png)",
                    f"{target_dir}/non_fire_images/": "Images WITHOUT fire (jpg/png)"
                }
            }
        )

    def _task():
        pipeline_state["running"] = True
        pipeline_state["status"]  = "retraining"
        pipeline_state["message"] = f"Retraining on {target_dir} …"
        TRAINING_STATUS.set(1)
        RETRAIN_COUNT.inc()
        try:
            _run_retrain_sync(target_dir)
            pipeline_state["status"]  = "completed"
            pipeline_state["message"] = "Retraining finished successfully."
        except Exception as exc:
            pipeline_state["status"]  = "failed"
            pipeline_state["message"] = str(exc)
            traceback.print_exc()
        finally:
            pipeline_state["running"] = False
            TRAINING_STATUS.set(0)
            push_metrics()

    background_tasks.add_task(_task)

    return {
        "message": "Retraining pipeline started in the background.",
        "dataset_used": target_dir,
        "status_endpoint": "/pipeline/status",
        "requirements": {
            "folder_structure": {
                "fire_images/":     "Images containing FIRE (JPEG/PNG, RGB)",
                "non_fire_images/": "Images WITHOUT fire (JPEG/PNG, RGB)"
            },
            "image_format":   "JPEG or PNG",
            "image_channels": "RGB (3 channels) – grayscale will fail",
            "image_resize":   "All images are auto-resized to 128×128 internally",
            "min_images":     "At least 50 images per class recommended"
        },
        "note": (
            "Retraining may take several minutes. "
            "Poll /pipeline/status for live updates."
        )
    }

class FeedbackRequest(BaseModel):
    feedback_id: str
    label: str

@app.post("/feedback", tags=["Inference"])
async def feedback(req: FeedbackRequest):
    """Accepts user feedback for low-confidence inferences, and stores augmented features."""
    if req.feedback_id not in pending_feedback:
        raise HTTPException(404, "Feedback ID not found or expired")
        
    timestamp, img_array = pending_feedback.pop(req.feedback_id)
    if time.time() - timestamp > FEEDBACK_TTL:
        raise HTTPException(404, "Feedback ID expired")
        
    if req.label not in CLASSES:
        raise HTTPException(400, f"Label must be one of {CLASSES}")
        
    save_feedback_features(req.feedback_id, req.label, img_array)
    return {"message": f"Feedback received. Augmented features saved for {req.label}."}

@app.post("/finetune", tags=["Pipeline"])
async def finetune(background_tasks: BackgroundTasks):
    """Starts finetuning the existing model using recently collected drift features."""
    if pipeline_state["running"]:
        raise HTTPException(status_code=409, detail="A pipeline run is already in progress.")
        
    if model_store["model"] is None:
        raise HTTPException(status_code=503, detail="No trained model found to finetune.")

    def _task():
        pipeline_state["running"] = True
        pipeline_state["status"]  = "finetuning"
        pipeline_state["message"] = "Finetuning on drift features..."
        TRAINING_STATUS.set(1)
        try:
            # We use the loaded model
            run_finetuning(model_store["model"])
            load_model_into_store()
            pipeline_state["status"]  = "completed"
            pipeline_state["message"] = "Finetuning finished successfully."
        except Exception as exc:
            pipeline_state["status"]  = "failed"
            pipeline_state["message"] = str(exc)
            traceback.print_exc()
        finally:
            pipeline_state["running"] = False
            TRAINING_STATUS.set(0)
            push_metrics()

    background_tasks.add_task(_task)
    
    return {
        "message": "Finetuning started in the background.",
        "status_endpoint": "/pipeline/status"
    }


# ── Status ────────────────────────────────────
@app.get("/pipeline/status", tags=["Pipeline"])
def pipeline_status():
    """Returns the current training / retraining status."""
    return {
        "running": pipeline_state["running"],
        "status":  pipeline_state["status"],
        "message": pipeline_state["message"],
        "model_loaded": model_store["model"] is not None
    }
