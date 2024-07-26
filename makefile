# Include .env file
include .env
export

.PHONY: run-inference run-inference-debug run-inference-test run-inference-test-debug run-proxy run-proxy-debug run-proxy-test run-proxy-test-debug

# Inference Service
run-inference:
	docker build --build-arg DD_API_KEY=$(DD_API_KEY) -t inference -f docker/Dockerfile .
	docker run --rm \
		--name inference \
		-p 8000:8000 \
		--runtime nvidia \
		--gpus 1 \
		-e APP_NAME=inference \
		-e DD_API_KEY=$(DD_API_KEY) \
		-e DD_SITE=us5.datadoghq.com \
		-e DD_ENV=production \
		-e DD_SERVICE=demo-inference \
		-e DD_VERSION=1.0.0 \
		-e DD_LOGS_INJECTION=true \
		-e DD_TRACE_ANALYTICS_ENABLED=true \
		-e DD_PROFILING_ENABLED=true \
		-e DD_TRACE_AGENT_URL=https://trace.agent.us5.datadoghq.com \
		inference \
		python -m gunicorn inference:app --worker-class uvicorn.workers.UvicornWorker --workers 1 --bind 0.0.0.0:8000


run-inference-debug:
	docker build --build-arg DD_API_KEY=$(DD_API_KEY) -t inference-debug -f docker/Dockerfile .
	docker run --rm \
		--name inference-debug \
		-p 8000:8000 -p 5678:5678 \
		--runtime nvidia \
		--gpus 1 \
		-e APP_NAME=inference \
		-e DD_API_KEY=None \
		-e DD_SITE=us5.datadoghq.com \
		-e DD_ENV=local \
		-e DD_SERVICE=demo-inference \
		-e DD_VERSION=1.0.0 \
		-e DD_LOGS_INJECTION=true \
		-e DD_TRACE_ANALYTICS_ENABLED=true \
		-e DD_PROFILING_ENABLED=true \
		-e DD_TRACE_AGENT_URL=https://trace.agent.us5.datadoghq.com \
		inference-debug \
		python -X frozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m gunicorn inference:app --worker-class uvicorn.workers.UvicornWorker --workers 1 --bind 0.0.0.0:8000


run-inference-test:
	docker build --build-arg DD_API_KEY=test -t inference-test -f docker/Dockerfile .
	docker run --rm \
		--runtime nvidia \
		--gpus 1 \
		-e DISABLE_TELEMETRY=True \
		-e DD_API_KEY=$(DD_API_KEY) \
		inference-test \
		python -m unittest discover -s tests

run-inference-test-debug:
	docker build --build-arg DD_API_KEY=test -t inference-test -f docker/Dockerfile .
	docker run -it --rm \
		-v $$(pwd):/workspace \
		-e DISABLE_TELEMETRY=True \
		-e DD_API_KEY=None \
		-p 5678:5678 \
		--runtime nvidia \
		--gpus 1 \
		inference-test \
		python -X frozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m unittest discover -s tests

# Proxy Service
run-proxy:
	docker build -t proxy -f docker/Dockerfile.proxy .
	docker run --rm \
		--name proxy \
		-p 8000:8000 \
		proxy \
		gunicorn proxy:app --worker-class uvicorn.workers.UvicornWorker --workers 1 --bind 0.0.0.0:8000

run-proxy-debug:
	docker build -t proxy-debug -f docker/Dockerfile.proxy .
	docker run --rm \
		--name proxy-debug \
		-p 8000:8000 -p 5679:5679 \
		proxy-debug \
		python -X frozen_modules=off -m debugpy --listen 0.0.0.0:5679 --wait-for-client -m gunicorn proxy:app --worker-class uvicorn.workers.UvicornWorker --workers 1 --bind 0.0.0.0:8000

test-cicd:
	@docker build -t test-cicd -f docker/Dockerfile . && \
	docker run --rm test-cicd python -m unittest discover -s tests

clean:
	@docker system prune -a --force

setup:
	@echo "Setting up model and dataset download environment..."
	@mkdir -p models/nateraw/food food101_data
	@chmod -R 777 models food101_data

	@echo "Building Docker image..."
	@docker build --build-arg UID=$$(id -u) --build-arg GID=$$(id -g) -t food101-downloader -f docker/Dockerfile.setup . || \
		(echo "Error: Docker build failed"; exit 1)

	@echo "Running model and dataset download container..."
	@docker run --rm \
		-v $$(pwd)/models:/workspace/models \
		-v $$(pwd)/food101_data:/workspace/food101_data \
		food101-downloader

	@echo "Verifying downloads..."
	@if [ ! -f "models/nateraw/food/pytorch_model.bin" ]; then \
		echo "Error: Model file is missing. Setup failed."; \
		exit 1; \
	fi
	@if [ ! -f "food101_data/data/train-00000-of-00008.parquet" ]; then \
		echo "Warning: Some dataset files might be missing. Check the food101_data directory."; \
	fi

	@echo "Fixing permissions..."
	@sudo chown -R $$(id -u):$$(id -g) models food101_data
	@chmod -R 755 models food101_data

	@echo "Setup complete. Check the 'models/nateraw/food' and 'food101_data' directories for downloaded files."

.PHONY: setup

# Run precommit on any staged files
precommit:
	@docker build --progress plain --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -f docker/Dockerfile.precommit -t precommit . && \
	docker run --rm -v $(shell pwd):/workspace precommit bash -c "pre-commit run"

# Run precommit on the entire repository
precommit-all:
	@docker build --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -f docker/Dockerfile.precommit -t precommit-all . && \
	docker run --rm -v $(shell pwd):/workspace precommit-all bash -c "pre-commit run --all-files"
