import json
import os
import subprocess
import time
import unittest

import requests


class TestGunicornCUDAInference(unittest.TestCase):
    server_process: subprocess.Popen | None = None

    @classmethod
    def setUpClass(cls) -> None:
        """Start the server with CUDA environment."""
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",  # Ensure CUDA is available
                "DD_ENV": "test",
                "DD_SERVICE": "inference-test",
                "DD_VERSION": "1.0.0",
            }
        )
        cls.server_process = subprocess.Popen(
            [
                "gunicorn",
                "inference:app",
                "--worker-class",
                "uvicorn.workers.UvicornWorker",
                "--workers",
                "2",
                "--bind",
                "0.0.0.0:8000",
            ],
            env=env,
            cwd="/workspace",  # Set the working directory to /workspace
        )
        time.sleep(10)  # Wait for the server to start and load the model

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop the server."""
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()

    def test_classify_endpoint(self) -> None:
        """Make a call to the food classification endpoint."""
        sample_image_path = "/workspace/tests/french_toast.jpeg"

        if not os.path.exists(sample_image_path):
            self.fail(f"Sample image not found at {sample_image_path}")

        with open(sample_image_path, "rb") as img_file:
            img_data = img_file.read()

        url = "http://0.0.0.0:8000/classify/"
        files = {"file": ("image.jpg", img_data, "image/jpeg")}

        try:
            response = requests.post(url, files=files)
        except requests.RequestException as e:
            self.fail(f"Request failed: {str(e)}")

        self.assertEqual(
            response.status_code,
            200,
            f"Expected status code to be 200 OK, got {response.status_code}",
        )

        try:
            data = response.json()
        except json.JSONDecodeError:
            self.fail(f"Failed to decode JSON response: {response.text}")

        self.assertIn("predictions", data, "Expected 'predictions' in the response")
        predictions = data["predictions"]
        self.assertGreater(len(predictions), 0, "Expected at least one prediction")

        # Assertions for French toast image
        top_prediction = predictions[0]
        self.assertEqual(
            top_prediction["label"], "french_toast", "Top prediction should be 'french_toast'"
        )
        self.assertGreater(
            top_prediction["score"], 0.9, "Confidence for French toast should be > 0.9"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
