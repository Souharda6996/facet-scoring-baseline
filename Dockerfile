FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/raw/ data/raw/
COPY eval/ eval/
COPY hallucination_demo/ hallucination_demo/
COPY tests/ tests/

# Preprocessing + embedding index have no external-service dependency and can
# run at image-build time so the image ships with data/processed/ populated.
RUN python src/preprocess.py && python src/embed_index.py

# Scoring (src/score.py, eval/run_eval.py) calls out to an Ollama server on
# the host/network at OLLAMA_URL (see src/llm_client.py) -- Ollama itself is
# not bundled in this image. Run Ollama separately (e.g. `ollama serve` on
# the host, or a sibling container) and point this container at it:
#   docker run --network host -e OLLAMA_URL=http://localhost:11434/api/chat facet-scorer
ENV OLLAMA_URL=http://host.docker.internal:11434/api/chat

CMD ["python", "-m", "pytest", "tests/", "-v"]
