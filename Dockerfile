FROM python:3.11-slim

WORKDIR /app

# sentence-transformers pulls in torch, and pip's default wheel is CUDA-enabled
# (multi-GB) even though this container never touches a GPU -- the actual LLM
# inference happens in Ollama, outside this image (see OLLAMA_URL below); this
# container only runs the small MiniLM embedding model on CPU. Installing the
# CPU-only torch wheel first (from PyTorch's own CPU index) means the later
# `pip install -r requirements.txt` finds torch already satisfied and skips
# the CUDA download entirely -- found while actually running `docker build`
# for the first time and watching it pull ~16GB of CUDA libraries for no
# reason (see DEBUGGING.md).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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
