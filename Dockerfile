# Minimal image for serving predictions. Training is NOT run at build
# time — the model is a generated artifact, mounted in via a volume, so
# retraining doesn't require rebuilding the image.

FROM python:3.11-slim

WORKDIR /app

# Copy dependency + source metadata first so Docker can cache the
# `pip install` layer when only application logic (not dependencies)
# changes on a later build.
COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

# Now copy everything else (scripts, README, etc.) — data/models/logs are
# excluded via .dockerignore and mounted as volumes instead.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "ml_project.api:app", "--host", "0.0.0.0", "--port", "8000"]
