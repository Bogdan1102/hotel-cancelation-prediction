# syntax=docker/dockerfile:1

# Base image — Python 3.11 slim (smaller than full python:3.11)
FROM python:3.11-slim

# Make Python logging visible in docker logs
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Working directory
WORKDIR /app

# Copy dependency list first (better layer caching)
# We support both Pipfile (preferred, cap. 5.3.1) and requirements.txt (fallback)
COPY Pipfile Pipfile.lock requirements.txt ./

# Install pipenv, then install dependencies from the lock file (cap. 5.3.1 + 5.3.2)
# --deploy + --system: skip virtualenv, install into system python
RUN pip install --no-cache-dir pipenv && \
    pipenv install --deploy --system && \
    rm -rf /root/.cache

# Copy the application code + pickled model + sample request
COPY train.py predict.py predict-test.py serve.py model.bin sample_request.json ./

# Expose the Flask port
EXPOSE 9696

# Run with gunicorn (production WSGI) instead of Flask dev server (cap. 5.3.1)
# serve:app references the Flask app object `app` inside serve.py
CMD ["gunicorn", "--bind=0.0.0.0:9696", "--workers=2", "serve:app"]
