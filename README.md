# EdgeGuard AI

A production-grade, edge-optimised fire detection system built with a complete MLOps pipeline. The system trains a lightweight MobileNet-style convolutional neural network on a binary fire/no-fire image dataset, exports the model to INT8 TFLite format for edge deployment, exposes a REST API for inference and retraining, and ships full observability through MLflow experiment tracking, Prometheus metrics, and a pre-built Grafana dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
  - [Step 1 - Clone the repository](#step-1---clone-the-repository)
  - [Step 2 - Set up the Python environment](#step-2---set-up-the-python-environment)
  - [Step 3 - Prepare the dataset](#step-3---prepare-the-dataset)
  - [Step 4 - Start the monitoring stack](#step-4---start-the-monitoring-stack)
  - [Step 5 - Run the training pipeline](#step-5---run-the-training-pipeline)
  - [Step 6 - Start the REST API](#step-6---start-the-rest-api)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
  - [Design Philosophy](#design-philosophy)
  - [Layer-by-Layer Breakdown](#layer-by-layer-breakdown)
  - [Hyperparameters](#hyperparameters)
  - [Data Augmentation](#data-augmentation)
  - [Post-Training Quantisation](#post-training-quantisation)
- [Pipeline Stages](#pipeline-stages)
- [REST API Reference](#rest-api-reference)
- [Monitoring and Observability](#monitoring-and-observability)
  - [MLflow Experiment Tracking](#mlflow-experiment-tracking)
  - [Prometheus Metrics](#prometheus-metrics)
  - [Grafana Dashboard](#grafana-dashboard)
- [CI/CD Workflows](#cicd-workflows)
- [Metrics Reference](#metrics-reference)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

Fire detection at the edge demands a model that is simultaneously accurate and small enough to run on resource-constrained hardware such as Raspberry Pi, Jetson Nano, or mobile SoCs. EdgeGuard AI solves this by implementing a custom depthwise separable convolution network (the same family of operations used in MobileNet) with an additional width multiplier that aggressively reduces parameter count without meaningful accuracy loss.

The complete system covers every phase of a production MLOps lifecycle:

- Automated, reproducible data ingestion with cardinality validation
- On-the-fly data augmentation baked into the model graph
- Configurable training with full per-epoch MLflow tracking
- Statistical performance analysis comparing the trained model against baseline architectures
- INT8 quantisation for embedded deployment
- A FastAPI inference server with built-in Prometheus instrumentation and asynchronous request logging
- Active Data Drift handling via a low-confidence feedback loop, automatic data augmentation, and targeted model finetuning
- A pre-provisioned Grafana dashboard with configured visual and Prometheus-level alerts for model decay and API errors

---

## Repository Structure

```
EdgeGuard-AI/
|
|-- config/
|   `-- config.yaml              Central configuration file for all hyperparameters and service URLs
|
|-- fire_dataset/
|   |-- fire_images/             Place fire images here (.jpg, .jpeg, .png)
|   `-- non_fire_images/         Place non-fire images here (.jpg, .jpeg, .png)
|
|-- models/
|   |-- fire_model.keras         Saved Keras model (generated after training)
|   `-- fire_model_int8.tflite   INT8 quantised TFLite model (generated after training)
|
|-- monitoring/
|   |-- prometheus.yml           Prometheus scrape configuration
|   `-- grafana/
|       `-- provisioning/
|           |-- datasources/
|           |   `-- prometheus.yml   Auto-configures Prometheus as Grafana data source
|           `-- dashboards/
|               |-- dashboard.yml             Grafana dashboard provider config
|               `-- edgeguard_dashboard.json  Pre-built dashboard (auto-loaded)
|
|-- src/
|   |-- utils.py                 Project root resolution and config loader
|   |-- data_ingestion.py        Dataset scanning, label alignment, stats reporting
|   |-- data_transformation.py   Image loading, resizing, normalisation, augmentation
|   |-- model_factory.py         Model architecture definition
|   |-- model_trainer.py         Training loop with MLflow integration
|   |-- model_evaluation.py      Test set evaluation, latency, throughput measurement
|   |-- model_pusher.py          TFLite INT8 conversion and export
|   |-- performance_analysis.py  Statistical comparison against baseline models
|   |-- monitoring_service.py    Pushes model metrics to Prometheus Pushgateway
|   |-- pipeline.py              Orchestrates all pipeline stages end-to-end
|   `-- api.py                   FastAPI application with inference and training endpoints
|
|-- .github/
|   `-- workflows/
|       |-- linting.yaml         Runs flake8 on every push and pull request
|       `-- training.yaml        Manually triggered pipeline run on GitHub Actions
|
|-- docker-compose.yml           Starts MLflow, Prometheus, Pushgateway, and Grafana
|-- requirements.txt             Python dependencies
|-- train_eval.py                Standalone training script (no pipeline infrastructure needed)
`-- README.md                    This file
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 4 GB | 8 GB |
| Disk space | 2 GB | 5 GB |
| Docker | 20.x | Latest |
| Docker Compose | 2.x | Latest |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Any |
| GPU | Not required | CUDA-capable GPU for faster training |

---

## Quick Start

Follow these steps in order. Each step is self-contained and explains exactly what you are doing and why.

### Step 1 - Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/HasnainRzza/EdgeGuard-AI.git
cd EdgeGuard-AI
```

### Step 2 - Set up the Python environment

It is strongly recommended to use a virtual environment so that the project's dependencies do not interfere with anything else installed on your machine.

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs TensorFlow, OpenCV, scikit-learn, MLflow, Prometheus client, FastAPI, and all other dependencies listed in `requirements.txt`.

### Step 3 - Prepare the dataset

The model expects images organised into exactly two folders inside the `fire_dataset/` directory. Place your images as follows:

```
fire_dataset/
|-- fire_images/       <-- images that CONTAIN fire (JPG or PNG)
`-- non_fire_images/   <-- images that DO NOT contain fire (JPG or PNG)
```

Rules for the dataset:
- Only JPEG (`.jpg`, `.jpeg`) and PNG (`.png`) files are processed. Any other files, including hidden files such as `.gitkeep`, are automatically ignored.
- The pipeline validates that the number of loaded images and labels are exactly equal before training begins. If any image fails to load (corrupted file, wrong format), it is silently skipped along with its label.
- A minimum of 50 images per class is recommended for meaningful training. More data will always produce better results.

You can download a public fire detection dataset from sources such as Kaggle (search for "fire detection dataset") or use your own images. After placing images in the folders, the pipeline will report the exact count loaded from each class when it runs.

### Step 4 - Start the monitoring stack

This project uses Docker to run four services: MLflow (experiment tracking), Prometheus (metrics collection), Pushgateway (metric ingestion from short-lived jobs), and Grafana (dashboard visualisation). Docker handles downloading the images and starting everything automatically.

Make sure Docker Desktop is running, then execute:

```bash
docker-compose up -d
```

The `-d` flag runs the containers in the background. After a few seconds the following services will be available:

| Service | URL | Default Credentials |
|---|---|---|
| MLflow UI | http://localhost:5000 | None required |
| Prometheus | http://localhost:9090 | None required |
| Pushgateway | http://localhost:9091 | None required |
| Grafana | http://localhost:3000 | admin / admin |

To stop all services later, run `docker-compose down`.

### Step 5 - Run the training pipeline

The training pipeline performs all steps automatically: it loads the dataset, transforms images, trains the model, evaluates it on the test set, pushes metrics to Prometheus, and exports a TFLite model. All training runs are recorded in MLflow.

Navigate to the `src/` directory and run:

```bash
cd src
python pipeline.py
```

You will see output similar to the following:

```
Dataset Stats:
  - Fire images (Class 0): 500
  - Non-Fire images (Class 1): 700
Transforming data...
Building model...
Training...
Epoch 1/15 - loss: 0.6821 - accuracy: 0.5943 - val_loss: 0.6512 - val_accuracy: 0.6120
...
Epoch 15/15 - loss: 0.1832 - accuracy: 0.9412 - val_loss: 0.2014 - val_accuracy: 0.9230
Test Accuracy: 0.9380
Inference Latency: 2.31 ms/image
Throughput: 432.87 images/sec
Converting to TFLite INT8...
TFLite model saved at models/fire_model_int8.tflite
Metrics pushed to Prometheus at localhost:9091
Pipeline completed successfully!
```

After training completes:
- `models/fire_model.keras` contains the full Keras model
- `models/fire_model_int8.tflite` contains the compressed INT8 model ready for embedded deployment
- The MLflow UI at http://localhost:5000 will show a new run under the experiment `EdgeGuard-AI-Training` with all hyperparameters and per-epoch metrics logged

### Step 6 - Start the REST API

The inference API allows you to submit images for fire detection and trigger retraining, all through standard HTTP requests.

From the `src/` directory, run:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag makes the server automatically restart when you change source files, which is useful during development. For a production deployment, omit `--reload`.

Once running, open your browser and navigate to:

- **http://localhost:8000/docs** - Interactive API documentation (Swagger UI). You can test every endpoint directly from your browser without writing any code.
- **http://localhost:8000** - Root endpoint listing all available routes

---

## Configuration

All configurable parameters live in a single file: `config/config.yaml`. You do not need to modify any Python source files to change how the model trains.

```yaml
artifacts_dir: "artifacts"
dataset_path: "fire_dataset"

training:
  img_size: 128        # Input image resolution (width and height in pixels)
  alpha: 0.5           # Width multiplier - controls model size vs accuracy trade-off
  classes: ["fire", "no_fire"]
  epochs: 15           # Number of complete passes through the training data
  batch_size: 16       # Number of images processed per gradient update
  learning_rate: 0.001 # Step size for the Adam optimiser

mlflow:
  experiment_name: "EdgeGuard-AI-Training"
  tracking_uri: "http://localhost:5000"

monitoring:
  pushgateway_url: "localhost:9091"

logging:
  batch_size: 20
  time_limit_sec: 5.0
  log_file: "logs/api_requests.jsonl"
```

**Width multiplier (`alpha`)** is the most impactful setting. It scales the number of filters in every convolutional layer by the given factor:
- `alpha: 1.0` - full-size model, highest accuracy, largest file size
- `alpha: 0.75` - 75% of full channels, good balance
- `alpha: 0.5` - 50% of full channels (default), edge-optimised
- `alpha: 0.25` - very small model, fastest inference, some accuracy loss

---

## Model Architecture

### Design Philosophy

The model is a custom depthwise separable convolutional network. Standard convolution applies a single filter across all input channels simultaneously, which is computationally expensive. Depthwise separable convolution splits this operation into two steps:

1. **Depthwise convolution**: A single filter is applied to each input channel independently. This captures spatial features.
2. **Pointwise convolution** (1x1 Conv2D): A 1x1 convolution combines information across channels. This captures cross-channel relationships.

This decomposition reduces the number of multiply-accumulate operations by a factor of approximately `1/N + 1/D^2`, where N is the number of output channels and D is the kernel size. For a 3x3 kernel this is roughly an 8-9x reduction in computation compared to standard convolution.

The width multiplier `alpha` then further reduces every filter count by a constant factor, making the model footprint configurable without changing the architecture code.

ReLU6 (clamped ReLU with a maximum value of 6) is used as the activation function throughout. This bounded activation behaves better than standard ReLU under fixed-point arithmetic, which is important for the INT8 quantisation step.

### Layer-by-Layer Breakdown

The network processes a single RGB image of shape `(128, 128, 3)`:

```
Input                   (128, 128, 3)
Data Augmentation       RandomFlip, RandomRotation, RandomZoom
Conv2D                  32*alpha filters, 3x3 kernel, stride 2   -> (64, 64, 16)
BatchNormalization
ReLU6

DepthwiseBlock 1        target filters: 32*alpha                  -> (64, 64, 16)
  DepthwiseConv2D 3x3, stride 1
  BatchNormalization + ReLU6
  Conv2D 1x1
  BatchNormalization + ReLU6

DepthwiseBlock 2        target filters: 64*alpha, stride 2        -> (32, 32, 32)
DepthwiseBlock 3        target filters: 64*alpha                  -> (32, 32, 32)
DepthwiseBlock 4        target filters: 128*alpha, stride 2       -> (16, 16, 64)
DepthwiseBlock 5        target filters: 128*alpha                 -> (16, 16, 64)
DepthwiseBlock 6        target filters: 256*alpha, stride 2       -> (8, 8, 128)
DepthwiseBlock 7        target filters: 256*alpha                 -> (8, 8, 128)

Dropout (rate=0.3)
GlobalAveragePooling2D                                            -> (128,)
Dense (2, softmax)                                               -> (2,)

Output: [P(fire), P(no_fire)]
```

With `alpha=0.5`, the model has approximately **40,000 to 50,000 trainable parameters**, compared to over 58 million in a VGG-style network solving the same problem.

### Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| Image size | 128 x 128 | Spatial input resolution |
| Width multiplier (alpha) | 0.5 | Channel scaling factor |
| Epochs | 15 | Training iterations over the full dataset |
| Batch size | 16 | Samples per gradient update |
| Learning rate | 0.001 | Adam optimiser initial step size |
| Optimiser | Adam | Adaptive moment estimation |
| Loss function | Sparse categorical crossentropy | Standard multi-class loss, accepts integer labels |
| Dropout rate | 0.3 | Fraction of units dropped during training to reduce overfitting |
| Activation | ReLU6 | Clamped ReLU, quantisation-friendly |
| Output activation | Softmax | Converts logits to class probabilities |
| Train / Val / Test split | 70% / 15% / 15% | Stratified split preserving class balance |

### Data Augmentation

Augmentation is applied as the first layer of the model graph itself (not as a separate preprocessing step). This means augmentation runs on the GPU when available and is automatically disabled during inference.

| Augmentation | Setting |
|---|---|
| Random horizontal flip | Applied with 50% probability |
| Random rotation | Up to 10% of a full rotation (~36 degrees) |
| Random zoom | Up to 10% zoom in or out |

### Post-Training Quantisation

After training, the model is converted from 32-bit floating point weights to 8-bit integer representation using TensorFlow Lite's full INT8 quantisation mode.

The conversion uses a representative dataset of 100 training samples to calibrate the integer scale factors for every layer. Both input and output tensors are set to `uint8`, making the model fully compatible with microcontrollers and edge accelerators that do not support floating-point operations.

| Property | Keras Model | TFLite INT8 |
|---|---|---|
| Weight precision | float32 | int8 |
| Size reduction | baseline | approximately 4x smaller |
| Inference target | GPU / CPU server | Raspberry Pi, microcontrollers, edge TPU |
| File | `models/fire_model.keras` | `models/fire_model_int8.tflite` |

---

## Pipeline Stages

Running `python src/pipeline.py` executes the following stages in order:

**Stage 1: Data Ingestion** (`data_ingestion.py`)
Scans `fire_images/` and `non_fire_images/` folders. Filters out hidden files and any file without a valid image extension (.jpg, .jpeg, .png, .bmp, .webp). Returns aligned NumPy arrays of file paths and integer labels (0 for fire, 1 for no-fire). Prints per-class counts.

**Stage 2: Train / Val / Test Split** (`pipeline.py`)
Splits image paths and labels using a stratified 70/15/15 ratio with a fixed random seed (42) for reproducibility. Stratification ensures both splits contain approximately the same proportion of fire and non-fire images.

**Stage 3: Data Transformation** (`data_transformation.py`)
Loads each image from disk using OpenCV, resizes to 128x128, converts from BGR to RGB colour space, and normalises pixel values to the range [0.0, 1.0]. If any image fails to load (corrupted file, disk error), it is removed from both X and y simultaneously, preserving cardinality. A final assertion confirms `len(X) == len(y)` before returning.

**Stage 4: Model Build** (`model_factory.py`)
Instantiates the architecture described above using the hyperparameters from `config.yaml`. Compiles with Adam and sparse categorical crossentropy.

**Stage 5: Training** (`model_trainer.py`)
Fits the model for the configured number of epochs. Logs all training and validation metrics (accuracy, loss, val_accuracy, val_loss) per epoch to MLflow. Logs all hyperparameters and system information (OS, RAM, CPU) as MLflow parameters. Saves the trained model to `models/fire_model.keras`.

**Stage 6: Evaluation** (`model_evaluation.py`)
Evaluates the saved model on the held-out test set. Measures inference latency per image and throughput in images per second. Prints a full classification report showing precision, recall, and F1-score for each class. Logs test metrics to MLflow.

**Stage 7: Performance Analysis** (`performance_analysis.py`)
Computes parameter reduction percentage and compression factor against two baseline architectures: a VGG-style network (~58M parameters) and MobileNet V1 baseline (~676K parameters). Reports accuracy per million parameters as an efficiency metric.

**Stage 8: Prometheus Monitoring** (`monitoring_service.py`)
Pushes model accuracy, loss, latency, throughput, and parameter count to the Prometheus Pushgateway under the job name `edgeguard_pipeline`.

**Stage 9: TFLite Conversion** (`model_pusher.py`)
Converts the Keras model to INT8 TFLite format and saves it to `models/fire_model_int8.tflite`.

---

## REST API Reference

Start the API server from the `src/` directory:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The interactive documentation at **http://localhost:8000/docs** lets you call every endpoint from your browser.

---

### GET /

Returns a summary of all available endpoints.

---

### GET /health

Returns the current health status of the service.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "pipeline_running": false
}
```

---

### POST /train

Starts the full training pipeline in the background. The API returns immediately; training continues asynchronously. You can monitor progress by polling `/pipeline/status`.

**Response:**
```json
{
  "message": "Training pipeline started in the background.",
  "monitor_at": "http://localhost:5000",
  "status_endpoint": "/pipeline/status"
}
```

If a training run is already in progress, the endpoint returns HTTP 409 Conflict.

---

### POST /infer

Runs fire detection on a single uploaded image.

**Request:** Multipart form upload with field name `file`. The file must be JPEG or PNG.

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

**Response:**
```json
{
  "prediction": "fire",
  "confidence": 97.43,
  "all_scores": {
    "fire": 97.43,
    "no_fire": 2.57
  },
  "latency_ms": 3.812,
  "image_resized_to": "128x128"
}
```

If no model has been trained yet, the endpoint returns HTTP 503 Service Unavailable.

**Note on Data Drift:** If the model's confidence is below 80%, the response will include `"needs_feedback": true` and a `"feedback_id"`. This image's preprocessed matrix is cached in memory for 5 minutes. You can submit the true label to the `/feedback` endpoint to store this data for future finetuning.

---

### POST /feedback

Accepts ground-truth labels for low-confidence inferences, automatically applies augmentation, and saves the features locally as `.npy` matrices to prevent data drift without storing raw images.

**Request Body (JSON):**
```json
{
  "feedback_id": "uuid-string-from-infer",
  "label": "fire"
}
```

---

### POST /finetune

Triggers an isolated background training job that finetunes the currently loaded model exclusively on the drift features collected via the `/feedback` endpoint.

**Response:**
```json
{
  "message": "Finetuning started in the background.",
  "status_endpoint": "/pipeline/status"
}
```

---

### POST /retrain

Retrains the model on a new dataset. Pass the absolute path to the new dataset directory as the `dataset_dir` query parameter. The directory must contain `fire_images/` and `non_fire_images/` subdirectories.

**Example:**
```
POST /retrain?dataset_dir=D:/my_new_fire_dataset
```

Leave `dataset_dir` empty to retrain on the default dataset path configured in `config.yaml`.

---

### GET /pipeline/status

Returns the current state of any running training or retraining job.

**Response:**
```json
{
  "running": false,
  "status": "completed",
  "message": "Training finished successfully.",
  "model_loaded": true
}
```

Possible status values: `idle`, `training`, `retraining`, `completed`, `failed`.

---

### GET /metrics

Returns all Prometheus metrics in the standard text exposition format. Prometheus scrapes this endpoint automatically based on the configuration in `monitoring/prometheus.yml`.

---

## Monitoring and Observability

### MLflow Experiment Tracking

MLflow records every training run, including hyperparameters, system information, and per-epoch metrics. Open the UI at **http://localhost:5000** to browse experiments, compare runs side by side, download model artifacts, and inspect the per-epoch learning curves.

The experiment is named `EdgeGuard-AI-Training`. Every run logs:

**Parameters logged per run:**
- `img_size`, `alpha`, `epochs`, `batch_size`, `learning_rate`, `classes`
- `platform`, `platform_release`, `architecture`, `processor`, `ram`, `python_version`
- `trainable_params`, `total_params`

**Metrics logged per epoch:**
- `accuracy`, `loss`, `val_accuracy`, `val_loss`

**Metrics logged at end of run:**
- `test_accuracy`, `test_loss`, `latency_ms`, `throughput`

### Prometheus Metrics

Two types of metrics flow into Prometheus:

**Model metrics** (pushed via Pushgateway after each training run, job=`edgeguard_pipeline`):

| Metric | Type | Description |
|---|---|---|
| `model_accuracy` | Gauge | Test set accuracy of the last trained model |
| `model_loss` | Gauge | Test set loss of the last trained model |
| `model_latency_ms` | Gauge | Per-image inference latency in milliseconds |
| `model_throughput_ips` | Gauge | Inference throughput in images per second |
| `model_parameters` | Gauge | Total parameter count of the last trained model |

**API metrics** (scraped directly from `/metrics` endpoint, job=`edgeguard_api`):

| Metric | Type | Description |
|---|---|---|
| `api_requests_total` | Counter | Total HTTP requests, labelled by method, endpoint, status code |
| `api_request_latency_seconds` | Histogram | End-to-end HTTP response latency per endpoint |
| `inference_total` | Counter | Total number of times the /infer endpoint was called |
| `inference_latency_seconds` | Histogram | Time spent inside the model's predict() call |
| `training_in_progress` | Gauge | 1 when a training run is active, 0 otherwise |
| `retrain_total` | Counter | Total number of retraining jobs triggered |

### Grafana Dashboard

The dashboard is provisioned automatically when Grafana starts. No manual import is required. When the Grafana container comes up, it reads the dashboard JSON from `monitoring/grafana/provisioning/dashboards/edgeguard_dashboard.json` and loads it immediately.

To view it:
1. Open **http://localhost:3000** in your browser
2. Log in with username `admin` and password `admin`
3. Click the grid icon (Dashboards) in the left sidebar
4. Click on **EdgeGuard AI Overview**

The dashboard contains the following panels:

| Panel | Type | What it shows |
|---|---|---|
| Model Accuracy | Stat | Latest test accuracy value from the last training run |
| Model Loss | Stat | Latest test loss from the last training run |
| Model Parameters | Stat | Total parameter count |
| Training In Progress | Stat | Live indicator: green when idle, highlights when training |
| API Request Traffic | Time series | Requests per second, broken down by endpoint and status code |
| API Latency | Time series | Average end-to-end response time per endpoint |
| Inference Operations | Time series | Volume of /infer calls over time |
| Inference Latency | Time series | Model-only computation time per inference call |

All panels query Prometheus and refresh automatically every 5 seconds. Visual color thresholds (green/orange/red) are embedded directly into panels to instantly flag degraded performance or high latency.

### System Alerts

System-level alert rules are defined in `monitoring/alert.rules` and natively integrated with Grafana's alerting engine. They include:
- **`ModelAccuracyDrop`**: Critical alert when accuracy falls below 80% (triggering the need for `/finetune`).
- **`HighInferenceLatency`**: Warning alert if average inference latency exceeds 1.0 second.
- **`HighAPIErrorRate`**: Critical alert if 5xx server errors spike.

### Asynchronous Request Logging

All API requests are logged via a non-blocking asynchronous batch logger. To ensure zero performance overhead on the inference endpoints, logs are queued in memory and flushed to disk (`logs/api_requests.jsonl`) by a background worker either when the batch size is reached (e.g. 20 requests) or the time limit expires (e.g. 5 seconds).

---

## CI/CD Workflows

Two GitHub Actions workflows are defined in `.github/workflows/`:

### Linting (triggers on every push and pull request to `main`)

File: `.github/workflows/linting.yaml`

Runs `flake8` against the `src/` directory and checks for syntax errors, undefined names, and invalid imports. This ensures the codebase is always in a parseable state regardless of environment.

```bash
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

### Model Training (manual trigger only)

File: `.github/workflows/training.yaml`

Triggered manually from the GitHub Actions tab (Actions > Model Training > Run workflow). Installs dependencies, runs the full pipeline, and uploads `models/fire_model_int8.tflite` as a downloadable workflow artifact. This workflow requires the dataset to be available in the repository or fetched via a separate data versioning step.

---

## Metrics Reference

The table below summarises all model performance metrics produced by the pipeline and where to find them.

| Metric | Definition | Where to find it |
|---|---|---|
| Training accuracy | Fraction of correctly classified training images per epoch | MLflow per-epoch chart, Grafana |
| Validation accuracy | Fraction of correctly classified validation images per epoch | MLflow per-epoch chart |
| Test accuracy | Accuracy on the held-out test set after training | MLflow summary, Grafana stat panel |
| Training loss | Sparse categorical crossentropy on training set per epoch | MLflow per-epoch chart |
| Validation loss | Sparse categorical crossentropy on validation set per epoch | MLflow per-epoch chart |
| Test loss | Final crossentropy on held-out test set | MLflow summary, Grafana stat panel |
| Inference latency (ms) | Average time to process one image, measured on test set | Terminal output, MLflow, Grafana |
| Throughput (images/sec) | Number of images processed per second on test set | Terminal output, MLflow |
| Precision | Of all images predicted as a class, how many were correct | Terminal classification report |
| Recall | Of all true images of a class, how many were identified | Terminal classification report |
| F1-score | Harmonic mean of precision and recall | Terminal classification report |
| Parameter count | Total number of scalar weights in the model | Terminal output, MLflow, Grafana |
| Compression vs VGG | Percentage reduction in parameters vs a 58M VGG-style model | Terminal performance analysis |
| Compression vs MobileNet | Percentage reduction vs the 676K MobileNet V1 baseline | Terminal performance analysis |
| Accuracy per million params | Test accuracy divided by parameter count in millions | Terminal performance analysis |

---

## Troubleshooting

**Problem: `ValueError: Data cardinality is ambiguous` during training**

This means the number of images loaded and the number of labels do not match. The most common cause is having non-image files (such as `.gitkeep` or `.DS_Store`) inside `fire_images/` or `non_fire_images/`. The pipeline in `src/` automatically filters these out. If you are using `train_eval.py` directly, ensure all files in the dataset folders are valid image files.

**Problem: `No trained model found at models/fire_model.keras`**

You need to run the training pipeline before starting the inference API. Follow Step 5 in the Quick Start section first, then start the API server.

**Problem: `Could not push to Prometheus: [Errno 111] Connection refused`**

The Prometheus Pushgateway is not running. Start the Docker stack with `docker-compose up -d` before running the pipeline.

**Problem: Docker containers fail to start**

Ensure Docker Desktop is running. On Windows, make sure WSL2 integration is enabled in Docker Desktop settings. Run `docker-compose logs` to see which container is failing and why.

**Problem: Grafana shows "No data" in panels**

This happens when no training run or API requests have occurred yet. After running the training pipeline and making at least one request to `/infer`, the panels will populate. Prometheus must also be running and scraping the Pushgateway and API server correctly. You can verify this at http://localhost:9090/targets.

**Problem: TensorFlow GPU warning on startup**

The message `GPU will not be used` is a warning, not an error. TensorFlow will fall back to CPU training automatically. Training will be slower but fully functional. To use a GPU on Windows, install the TensorFlow-DirectML plugin. On Linux, install CUDA and cuDNN matching your TensorFlow version.
