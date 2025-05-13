#!/bin/bash

# Path to the NVIDIA PyTorch image
NVIDIA_IMAGE="/dss/dsshome1/05/ge26sav2/autops-rl/nvidia+pytorch+25.02-py3.sqsh"

# Create the container if it doesn't exist
if [ ! -d "/raid/enroot/data/user-$USER/autops-rl" ]; then
    echo "Creating Enroot container..."
    enroot create --name autops-rl "$NVIDIA_IMAGE"
    
    # Start the container and install dependencies
    echo "Installing dependencies in the container..."
    enroot start --root --mount /dss/dsshome1/05/ge26sav2/autops-rl:/workspace autops-rl bash -c "
        # Install dependencies
        pip install --no-cache-dir -r /workspace/requirements.txt
        
        # Create a marker file to indicate dependencies are installed
        touch /workspace/.dependencies_installed
        
        echo 'Container setup complete with all dependencies installed!'
    "
else
    echo "Container already exists. Checking if dependencies are installed..."
    
    # Check if dependencies are installed
    if [ ! -f "/dss/dsshome1/05/ge26sav2/autops-rl/.dependencies_installed" ]; then
        echo "Installing dependencies in the existing container..."
        enroot start --root --mount /dss/dsshome1/05/ge26sav2/autops-rl:/workspace autops-rl bash -c "
            # Install dependencies
            pip install --no-cache-dir -r /workspace/requirements.txt
            
            # Create a marker file to indicate dependencies are installed
            touch /workspace/.dependencies_installed
            
            echo 'Dependencies installation complete!'
        "
    else
        echo "Dependencies already installed. Container is ready to use."
    fi
fi

echo "Container 'autops-rl' is ready to use!"