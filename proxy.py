"""Proxy for forwarding image classification requests to a more powerful server."""

import os
import time
from typing import List

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
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

patch_all()
config.env = os.getenv("DD_ENV", "production")
config.service = os.getenv("DD_SERVICE", "proxy")
config.version = os.getenv("DD_VERSION", "1.0.0")
Profiler().start()

dd_config = Configuration()
dd_config.server_variables["site"] = os.getenv("DD_SITE", "us5.datadoghq.com")
dd_config.api_key["apiKeyAuth"] = os.getenv("DD_API_KEY")
dd_config.api_key["appKeyAuth"] = os.getenv("DD_APP_KEY")

POD_ID = os.environ.get("HOSTNAME", "Unknown")


@app.post("/classify/")
async def proxy_classify(request: Request, files: List[UploadFile] = File(...)):
    """Forwards image classification requests onto the inference server."""
    print(f"Received request to /classify/ with {len(files)} files")
    for file in files:
        print(f"File: {file.filename}, Content-Type: {file.content_type}")

    inference_endpoint = request.headers.get("X-Inference-Endpoint")
    if not inference_endpoint:
        raise HTTPException(status_code=400, detail="X-Inference-Endpoint header is required")

    async with httpx.AsyncClient() as client:
        try:
            # Prepare files for the request
            files_data = [
                ("files", (file.filename, await file.read(), file.content_type)) for file in files
            ]

            # Forward the request to the inference server
            response = await client.post(inference_endpoint, files=files_data, timeout=60.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"HTTP Status Error: {e}")
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            print(f"Request Error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error requesting {inference_endpoint}: {str(e)}"
            )

    modified_content = {
        "pod_id": POD_ID,
        "original_response": response.json(),
        "inference_endpoint": inference_endpoint,
    }
    return JSONResponse(content=modified_content, status_code=response.status_code)


@app.get("/health")
async def health():
    """Keep track of service health."""
    print("Health check requested")
    metric = MetricPayload(
        series=[
            MetricSeries(
                metric="proxy.health_check",
                type=MetricIntakeType.COUNT,
                points=[MetricPoint(timestamp=int(time.time()), value=1)],
                tags=[
                    f"pod:{POD_ID}",
                    f"env:{config.env}",
                    f"service:{config.service}",
                    f"version:{config.version}",
                ],
            )
        ]
    )

    try:
        with ApiClient(dd_config) as api_client:
            response = MetricsApi(api_client).submit_metrics(body=metric)
        print(f"Metric submitted successfully. Response: {response}")
    except Exception as e:
        print(f"Error submitting metric to Datadog: {e}")
        print(f"Attempted to submit: {metric.to_dict()}")

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
