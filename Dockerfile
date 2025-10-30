FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./

# Pre-download VADER lexicon (safer syntax for Railway builds)
RUN python -m nltk.downloader vader_lexicon

CMD ["python", "main.py"]
