# Use the official Python 3.11 image from Docker Hub
FROM python:3.11.6-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the default command to an interactive shell
CMD ["/bin/bash"]