#!/bin/bash

# Source the .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found."
    exit 1
fi

# Check if DD_API_KEY is set
if [ -z "$DD_API_KEY" ]; then
    echo "Error: DD_API_KEY is not set in the .env file."
    exit 1
fi

# Run Docker commands
docker build --build-arg DD_API_KEY="$DD_API_KEY" -t registry.digitalocean.com/corelogic-cr-demo/inference:latest -f docker/Dockerfile .
docker push registry.digitalocean.com/corelogic-cr-demo/inference:latest
