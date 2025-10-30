FROM python:3.12-slim

WORKDIR /app

# System deps (build tools are small but help with some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY main.py ./

# (Optional) Pre-download the VADER lexicon to avoid first-run download
RUN python - <<'PY'\nimport nltk\nnltk.download('vader_lexicon')\nPY

CMD ["python", "main.py"]
