# Stage 1: Base image for general use with Python 3.11.6
FROM python:3.11.6-slim AS base

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Create a local directory for Python packages
RUN mkdir -p /app/python_packages

# Install the required Python packages for general use into the local directory
RUN pip install --no-cache-dir -r requirements.txt --target /app/python_packages

# Copy the rest of the application code into the container
COPY . .

# Set the PYTHONPATH environment variable to include the local packages directory
ENV PYTHONPATH="/app/python_packages:${PYTHONPATH}"

# Stage 2: NVIDIA Jetson specific image
FROM nvcr.io/nvidia/l4t-base:r35.1.0 AS jetson

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Create a local directory for Python packages
RUN mkdir -p /app/python_packages

# Remove torch from the requirements for Jetson and install the rest locally
RUN sed -i '/torch/d' requirements.txt && pip install --no-cache-dir -r requirements.txt --target /app/python_packages

# Copy the rest of the application code into the container
COPY . .

# Set the PYTHONPATH environment variable to include the local packages directory
ENV PYTHONPATH="/app/python_packages:${PYTHONPATH}"

# Final stage: Default to using the base image for general use
FROM base

# Set the default command to an interactive shell
CMD ["/bin/bash"]