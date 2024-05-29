# Stage 1: Base image for general use with Python 3.11.6
FROM python:3.11.6-slim AS base

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages for general use globally
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir --verbose

# Copy the rest of the application code into the container
COPY . .

# Stage 2: NVIDIA Jetson specific image
FROM nvcr.io/nvidia/pytorch:24.05-py3 AS jetson

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements-jetson.txt file into the container
COPY requirements-jetson.txt .

# Install the required Python packages for Jetson globally
RUN pip install --upgrade pip && \
    pip install -r requirements-jetson.txt --no-cache-dir --verbose

# Copy the rest of the application code into the container
COPY . .

# Final stage: Default to using the base image for general use
FROM base

# Set the working directory inside the container
# WORKDIR /app

# Copy the python packages from the Jetson stage to the base stage
# COPY --from=jetson /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Set the default command to an interactive shell
CMD ["/bin/bash"]