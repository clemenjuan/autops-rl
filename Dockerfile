# Stage 1: Base image for general use with Python 3.11.6
FROM python:3.11.6-slim AS base
 
# Set the working directory inside the container
WORKDIR /app
 
# Copy the requirements.txt file into the container
COPY requirements.txt .
 
# Create a local directory for Python packages
# RUN mkdir -p /app/python_packages
 
# Install the required Python packages for general use into the local directory
RUN pip install -r requirements.txt
 
# Copy the rest of the application code into the container
COPY . .
 
# Set the PYTHONPATH environment variable to include the local packages directory
# ENV PYTHONPATH="/app/python_packages:${PYTHONPATH}"
 
# Stage 2: NVIDIA Jetson specific image
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 AS jetson
 
# Install Python and pip
# RUN apt-get update && apt-get install -y python3 python3-pip
 
# Set the working directory inside the container
WORKDIR /app
 
# Copy the requirements.txt file into the container
COPY requirements-jetson.txt .
 
# Create a local directory for Python packages
# RUN mkdir -p /app/python_packages
 
# Remove torch from the requirements for Jetson and install the rest locally
RUN pip install -r requirements-jetson.txt

# Copy the rest of the application code into the container
COPY . .
 
# Set the PYTHONPATH environment variable to include the local packages directory
# ENV PYTHONPATH="/app/python_packages:${PYTHONPATH}"
 
# Final stage: Default to using the base image for general use
FROM base
 
# Set the default command to an interactive shell
CMD ["/bin/bash"]