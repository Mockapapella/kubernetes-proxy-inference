"""Proxy for forwarding image classification requests to a more powerful server."""

import os
import time

import httpx
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
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import JSONResponse

patch_all()

# Configure Datadog tracing
config.env = os.getenv("DD_ENV", "production")
config.service = os.getenv("DD_SERVICE", "proxy")
config.version = os.getenv("DD_VERSION", "1.0.0")

# Start the profiler
profiler = Profiler()
profiler.start()


app = FastAPI()

# Initialize Datadog configuration
configuration = Configuration()
configuration.server_variables["site"] = os.getenv("DD_SITE", "us5.datadoghq.com")
configuration.api_key["apiKeyAuth"] = os.getenv("DD_API_KEY")
configuration.api_key["appKeyAuth"] = os.getenv("DD_APP_KEY")

print("Datadog configuration:")
print(f"Site: {configuration.server_variables['site']}")
print(
    f"API Key: {'*' * len(configuration.api_key['apiKeyAuth']) if configuration.api_key['apiKeyAuth'] else 'Not set'}"
)
print(
    f"APP Key: {'*' * len(configuration.api_key['appKeyAuth']) if configuration.api_key['appKeyAuth'] else 'Not set'}"
)


@app.post("/classify/")
async def proxy_classify(request: Request, file: UploadFile = File(...)):
    """Forwards image classification requests onto a beefy GPU server."""
    inference_endpoint = "https://hqstk9y08by2lb-8001.proxy.runpod.net/classify/"

    # Read the file content
    file_content = await file.read()

    # Prepare the file for the POST request
    files = {"file": (file.filename, file_content, file.content_type)}

    async with httpx.AsyncClient() as client:
        response = await client.post(inference_endpoint, files=files)

    # Get the pod ID
    pod_id = os.environ.get("HOSTNAME", "Unknown")

    # Get the original response content
    original_content = response.json()

    # Add the pod ID to the response
    modified_content = {"pod_id": pod_id, "original_response": original_content}

    return JSONResponse(content=modified_content, status_code=response.status_code)


@app.get("/health")
async def health():
    """Keep track of service health."""
    pod_id = os.environ.get("HOSTNAME", "Unknown")

    # Send a custom metric to Datadog
    with ApiClient(configuration) as api_client:
        api_instance = MetricsApi(api_client)
        current_time = int(time.time())
        metric = MetricPayload(
            series=[
                MetricSeries(
                    metric="proxy.health_check",
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

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
