# MLOPS-17: the ml-inference FastAPI sidecar. Vendors the ml package so the TCN
# loads in-container; the trained model + metadata are mounted as a volume
# (editable without rebuilding) and pinned by models/manifest.json (MLOPS-05).
FROM python:3.12-slim
WORKDIR /app

COPY ml-inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The ml package (flash_crash_watchdog) and its inference entrypoint.
COPY ml /app/ml
RUN pip install --no-cache-dir -e /app/ml
COPY ml-inference/server.py /app/server.py

# Models + manifest are mount points.
ENV PYTHONPATH=/app/ml
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]