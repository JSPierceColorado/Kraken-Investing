FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# (Optional but recommended) upgrade pip to get latest wheels/resolver
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY main.py ./

# Pre-download VADER lexicon
RUN python -m nltk.downloader vader_lexicon

CMD ["python", "main.py"]
