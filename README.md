# Master Thesis Project

This repository contains the code for my master thesis project. The project is set up to run in a Docker container to ensure a consistent development environment across different platforms (macOS, Windows, Linux, NVIDIA Jetson).

## Prerequisites

Make sure you have Docker installed on your system. Follow the instructions below to install Docker on various platforms.

### Docker Installation

#### macOS or Windows

1. Download and install Docker Desktop from [here](https://www.docker.com/products/docker-desktop).
2. Follow the installation instructions and start Docker Desktop.

#### Linux

1. Update your package list and install Docker:

   ```sh
   sudo apt-get update
   sudo apt-get install -y docker.io
   ```

2. Start Docker and enable it to run at startup:

    ```sh
    sudo systemctl start docker
    sudo systemctl enable docker
    ```

3.	Add your user to the Docker group (optional but recommended):

    ```sh
    sudo usermod -aG docker $USER
    ```

Log out and log back in for the changes to take effect.
   


#### NVIDIA Jetson

Ensure you have Docker installed and set up on your Jetson device. Use NVIDIA Docker to run the container with GPU support:


1. Install Docker:

    ```sh
    sudo apt-get update
    sudo apt-get install -y docker.io
    ``` 

2. Verify CUDA installation:

    ```sh
    nvcc --version
    nvidia-smi
    ``` 


3.	Install NVIDIA Container Toolkit:

    ```sh
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ``` 

4.	Configure Docker to use NVIDIA runtime by creating or editing /etc/docker/daemon.json:

    ```json
    {
    "runtimes": {
        "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
    }
    ```

5. Then restart Docker:

    ```sh
    sudo systemctl restart docker
    ``` 

6. Pull the compatible PyTorch Docker image:

    ```sh
    docker pull nvcr.io/nvidia/l4t-base:r35.1.0
    ```



## Setting Up the Project

1. Clone the repository to your local machine:

   ```sh
   git clone https://gitlab.lrz.de/clemente.juan/masterthesis_git.git
   cd masterthesis_git
   ```

2. Build the Docker image (might need sudo)

- For general use:

    ```sh
    docker build -t masterthesis_clemente .
    ```

- For NVIDIA Jetson:

    ```sh
    sudo docker build --target jetson -t masterthesis_clemente:jetson .
    ``` 
    


### Running the Docker Container

To start an interactive session inside the Docker container, use the following commands based on your operating system. Change the --shm-size=xgb (shared memory) to make sure to set this to more than 30% of available RAM:

#### On macOS and Linux

```sh
docker run --rm -it --shm-size=2gb -v $(pwd):/app masterthesis_clemente
```


#### On Windows (Command Prompt or PowerShell)

```sh
docker run --rm -it --shm-size=2gb -v %cd%:/app masterthesis_clemente
```


#### On NVIDIA Jetson

```sh
sudo docker run --rm -it --shm-size=2gb --runtime=nvidia --gpus all -v $(pwd):/app masterthesis_clemente:jetson /bin/bash
``` 

And verify CUDA and PyTorch availability inside the container:

```sh
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
``` 

By following these steps, you can ensure that your Docker container on NVIDIA Jetson with JetPack 5.1.x has access to CUDA 11.4 and PyTorch 2.0.0.

### Running Your Scripts

Once inside the Docker container, you can run any of your Python scripts. For example:

#### Monte-Carlo simulation

No trained agent involved. Edit simulation parameters at the end of the file ```FSS_env.py``` .

```python
python3 FSS_env.py
```

#### Train Agents

Training for different policies (PPO, DQN, A2C, A3C, IMPALA). Edit the configuration setup at the beginning of training.py to include gpu resources, more satellites or add more parallelism to the training process. 
To perform hyperparameter tuning and training for different policies, run the following commands (customizables):

```python
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy ppo --checkpoint-dir ppo_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy dqn --checkpoint-dir dqn_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a2c --checkpoint-dir a2c_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a3c --checkpoint-dir a3c_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy impala --checkpoint-dir impala_checkpoints --tune
```

If you only want to train the policies without tuning, omit the --tune argument (also customizables):

```python
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy ppo --checkpoint-dir ppo_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy dqn --checkpoint-dir dqn_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a2c --checkpoint-dir a2c_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a3c --checkpoint-dir a3c_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy impala --checkpoint-dir impala_checkpoints
```

Finally, if you want to train from a previous checkpoint, run the following commands (of course customize them according to your needs):

```python
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy ppo --checkpoint-dir ppo_checkpoints/ppo_policy
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy dqn --checkpoint-dir dqn_checkpoints/dqn_policy
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a2c --checkpoint-dir a2c_checkpoints/a2c_policy
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a3c --checkpoint-dir a3c_checkpoints/a3c_policy
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy impala --checkpoint-dir impala_checkpoints/impala_policy
```


#### Test Trained Agents
To test the trained policies, you can run the following commands:

```python
python3 main.py --framework torch --policy ppo --checkpoint-dir ppo_checkpoints/ppo_policy
``` 




## Using Cloud Computing Platforms

### Google Colab

### AWS (Amazon Web Services)

#### Using EC2

#### Using AWS ECS

#### Using AWS Lambda

## Troubleshooting

## Contributing
If you have any suggestions or improvements, feel free to create a pull request or open an issue in this repository.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
