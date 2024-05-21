# Stage 1: Base image for general use with Python 3.11.6
FROM python:3.11.6-slim AS base

# Set the working directory inside the container
WORKDIR /app

# Copy the general use requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages for general use
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Stage 2: NVIDIA Jetson specific image
FROM nvcr.io/nvidia/pytorch:23.05-py3 AS jetson

# Set the working directory inside the container
WORKDIR /app

# Copy the Jetson specific requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages for Jetson
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Default to using the base image for general use
FROM base

# Set the default command to an interactive shell
CMD ["/bin/bash"]