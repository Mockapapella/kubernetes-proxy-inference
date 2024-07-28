"""AI Image classification API. Works on both GPU and CPU."""

import io
import logging
import os
import time

import torch
from datadog_api_client import ApiClient
from datadog_api_client import Configuration
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_series import MetricSeries
from ddtrace import patch_all
from ddtrace.profiling import Profiler
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from PIL import Image
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

patch_all()

# Configuration
MODEL_PATH = "/workspace/models/nateraw/food"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DD_ENV = os.getenv("DD_ENV", "production")
DD_SERVICE = os.getenv("DD_SERVICE", "inference")
DD_VERSION = os.getenv("DD_VERSION", "1.0.0")
DD_SITE = os.getenv("DD_SITE", "us5.datadoghq.com")
DD_API_KEY = os.getenv("DD_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize Datadog
Profiler().start()
dd_config = Configuration()
dd_config.server_variables["site"] = DD_SITE
dd_config.api_key["apiKeyAuth"] = DD_API_KEY

# Initialize model
logger.info(f"Initializing model from {MODEL_PATH} on device {DEVICE}")
food_classifier = pipeline("image-classification", model=MODEL_PATH, device=DEVICE)
logger.info("Model initialization complete")


def send_metric(metric_name, value, metric_type, tags=None):
    """Send a metric to Datadog."""
    with ApiClient(dd_config) as api_client:
        api_instance = MetricsApi(api_client)
        metric = MetricPayload(
            series=[
                MetricSeries(
                    metric=metric_name,
                    type=metric_type,
                    points=[MetricPoint(timestamp=int(time.time()), value=value)],
                    tags=tags or [],
                )
            ]
        )
        try:
            api_instance.submit_metrics(body=metric)
            logger.info(f"Sent metric {metric_name} to Datadog")
        except Exception as e:
            logger.error(f"Error sending metric to Datadog: {e}")


@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    """Classify food in the uploaded image."""
    logger.info(f"Received classification request for file: {file.filename}")
    if not file:
        logger.error("No file provided for classification")
        raise HTTPException(status_code=400, detail="No file provided for classification.")
    try:
        start_time = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = food_classifier(image)
        predictions = [
            {"label": result["label"], "score": float(result["score"])} for result in results
        ]

        # Calculate and send metrics
        process_time = time.time() - start_time
        send_metric(
            "inference.process_time",
            process_time,
            MetricIntakeType.GAUGE,
            [f"env:{DD_ENV}", f"model:{MODEL_PATH}"],
        )
        send_metric(
            "inference.requests",
            1,
            MetricIntakeType.COUNT,
            [f"env:{DD_ENV}", f"model:{MODEL_PATH}"],
        )

        logger.info(f"Successfully classified image. Top prediction: {predictions[0]['label']}")
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        send_metric(
            "inference.errors", 1, MetricIntakeType.COUNT, [f"env:{DD_ENV}", f"model:{MODEL_PATH}"]
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check for the service."""
    logger.info("Health check requested")
    pod_id = os.environ.get("HOSTNAME", "Unknown")

    send_metric(
        "inference.health_check",
        1,
        MetricIntakeType.COUNT,
        [f"pod:{pod_id}", f"env:{DD_ENV}", f"service:{DD_SERVICE}", f"version:{DD_VERSION}"],
    )

    return {"status": "healthy", "pod_id": pod_id, "model_device": DEVICE, "environment": DD_ENV}


if __name__ == "__main__":
    logger.info(f"Starting server in {DD_ENV} environment")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
