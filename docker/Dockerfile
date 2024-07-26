FROM python:3.11-bookworm

ARG DD_API_KEY

WORKDIR /workspace

# Install Datadog agent without starting it
RUN DD_API_KEY=${DD_API_KEY} DD_SITE="us5.datadoghq.com" DD_INSTALL_ONLY=true \
    bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script_agent7.sh)"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create entrypoint script
RUN echo '#!/bin/bash\n\
/opt/datadog-agent/bin/agent/agent start &\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["gunicorn", "inference:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "1", "--bind", "0.0.0.0:8000"]
