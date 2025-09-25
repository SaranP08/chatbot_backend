FROM python:3.13-slim

# Install Rust
RUN apt-get update && apt-get install -y curl build-essential \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && export PATH="$HOME/.cargo/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
