"""AI Image classification API. Works on both GPU and CPU."""

import io
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
from ddtrace import config
from ddtrace import patch_all
from ddtrace.profiling import Profiler
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification
from transformers import pipeline

patch_all()

# Configure Datadog tracing
config.env = os.getenv("DD_ENV", "production")
config.service = os.getenv("DD_SERVICE", "inference")
config.version = os.getenv("DD_VERSION", "1.0.0")

# Start the profiler
profiler = Profiler()
profiler.start()

# Setting up model pipeline
local_model_path = "/workspace/models/nateraw/food"
model = AutoModelForImageClassification.from_pretrained(local_model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(local_model_path)
device = 0 if torch.cuda.is_available() else -1
food_classifier = pipeline(
    "image-classification", model=model, feature_extractor=feature_extractor, device=device
)

app = FastAPI()

# Initialize Datadog configuration
configuration = Configuration()
configuration.server_variables["site"] = os.getenv("DD_SITE", "us5.datadoghq.com")
configuration.api_key["apiKeyAuth"] = os.getenv("DD_API_KEY")

print("Datadog configuration:")
print(f"Site: {configuration.server_variables['site']}")
print(
    f"API Key: {'*' * len(configuration.api_key['apiKeyAuth']) if configuration.api_key['apiKeyAuth'] else 'Not set'}"
)


@app.post("/classify/")
async def classify_food(file: UploadFile = File(...)):
    """Classify food in the uploaded image."""
    print(f"Received image: {file.filename}")
    if not file:
        raise HTTPException(status_code=400, detail="No file provided for classification.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        results = food_classifier(image)

        predictions = [
            {"label": result["label"], "score": float(result["score"])} for result in results
        ]

        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check for the service."""
    pod_id = os.environ.get("HOSTNAME", "Unknown")

    # Send a custom metric to Datadog
    with ApiClient(configuration) as api_client:
        api_instance = MetricsApi(api_client)
        current_time = int(time.time())
        metric = MetricPayload(
            series=[
                MetricSeries(
                    metric="inference.health_check",
                    type=MetricIntakeType.COUNT,
                    points=[MetricPoint(timestamp=current_time, value=1)],
                    tags=[
                        f"pod:{pod_id}",
                        f"env:{config.env}",
                        f"service:{config.service}",
                        f"version:{config.version}",
                    ],
                )
            ]
        )
        try:
            response = api_instance.submit_metrics(body=metric)
            print(f"Metric submitted successfully. Response: {response}")
        except Exception as e:
            print(f"Error submitting metric to Datadog: {e}")
            print(f"Attempted to submit: {metric.to_dict()}")

    return {"message": "Welcome to the Food Classification API", "pod_id": pod_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
