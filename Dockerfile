# The web app online (README "Run it online"). Serving needs no torch: only `run.py embed` uses it.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 TAGGER_HOST=0.0.0.0
WORKDIR /app
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision|open_clip_torch|timm|huggingface_hub)==' requirements.txt > serve.txt \
 && pip install --no-cache-dir -r serve.txt
COPY . .
# labels/ and data/ live in the Cloud Storage bucket mounted at /mnt/state, so edits survive restarts
RUN rm -rf data labels && ln -s /mnt/state/data data && ln -s /mnt/state/labels labels
CMD mkdir -p /mnt/state/data/image_vectors /mnt/state/labels && exec python run.py serve --port "$PORT"
