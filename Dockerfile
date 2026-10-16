FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY simulator simulator
COPY engine engine
COPY agents agents
COPY ui ui
COPY data/match_001/events.jsonl data/match_001/tracking.jsonl data/match_001/truth.json data/match_001/
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
CMD ["python", "-m", "ui.server", "--speed", "20"]
