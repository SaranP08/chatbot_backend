# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# --- THIS IS THE CRITICAL NEW SECTION ---
# Install system dependencies required for building llama-cpp-python
# - apt-get update: Refreshes the package list
# - build-essential: Installs C/C++ compilers (gcc, g++) and make
# - cmake: The build system generator used by llama-cpp-python
# - --no-install-recommends: Reduces image size by not installing optional packages
# - rm -rf /var/lib/apt/lists/*: Cleans up the apt cache to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*
# --- END OF NEW SECTION ---

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local 'data' directory into the container's /app/data directory
COPY ./data ./data

# Copy the rest of the application source code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:10000", "--timeout", "120", "main:app"]