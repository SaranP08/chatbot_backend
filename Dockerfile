# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port for Render
EXPOSE 8000

# Start command using Render PORT env variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
