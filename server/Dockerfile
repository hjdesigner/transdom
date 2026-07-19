FROM python:3.14-slim

WORKDIR /app

# Copy only the dependency list first — Docker caches this layer,
# so it only reinstalls packages when requirements.txt actually changes,
# not every time you edit main.py.
COPY requirements.txt .
# Use PyTorch's CPU-only wheel index — avoids downloading GPU/CUDA
# binaries we don't need, since this container has no GPU anyway.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]