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


2. Then restart Docker and verify configuration:

    ```sh
    sudo systemctl restart docker
    sudo docker info | grep "Docker Root Dir"
    ``` 




## Setting Up the Project

1. Clone the repository to your local machine:

   ```sh
   git clone https://gitlab.lrz.de/clemente.juan/masterthesis_git.git
   cd masterthesis_git
   ```

2. Build the Docker image (might need sudo):

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
docker run -e PYTHONWARNINGS="ignore::DeprecationWarning" --rm -it --shm-size=6gb -v $(pwd):/app masterthesis_clemente
```


#### On Windows (Command Prompt or PowerShell)

```sh
docker run -e PYTHONWARNINGS="ignore::DeprecationWarning" --rm -it --shm-size=6gb -v %cd%:/app masterthesis_clemente
```


#### On NVIDIA Jetson (with GPU support)

```sh
sudo docker run -e PYTHONWARNINGS="ignore::DeprecationWarning" -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia --network host -v $(pwd):/app masterthesis_clemente:jetsonv2 /bin/bash
``` 

And verify CUDA and PyTorch availability inside the container:

```sh
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available())"
``` 

By following these steps, you can ensure that your Docker container on NVIDIA Jetson with JetPack has access to CUDA and PyTorch.

### Running Your Scripts

Once inside the Docker container, you can run any of your Python scripts. For example:

#### Monte-Carlo simulation

No trained agent involved. Edit simulation parameters at the end of the file ```FSS_env.py``` .

```python
python3 FSS_env.py
```

#### Train Agents

Training for different policies (PPO, DQN, A2C, A3C, IMPALA) can be configured by editing the setup at the beginning of training.py. This includes specifying GPU resources, the number of satellites, or adding more parallelism to the training process.

##### Available arguments
When running `training.py`, you can customize the training process using the following arguments:

- `--framework`: Specifies the deep learning framework. Choices are `tf`, `tf2`, or `torch`. Default is `torch`.
- `--stop-iters`: The number of iterations to train. Default is `50`.
- `--stop-reward`: The reward threshold at which training stops. Default is `1000000.0`.
- `--policy`: The policy to train. Choices are `ppo`, `dqn`, `a2c`, `a3c`, or `impala`. Default is `ppo`.
- `--checkpoint-dir`: Directory to save checkpoints. Default is `checkpoints`.
- `--tune`: Whether to perform hyperparameter tuning. This is a flag argument.
- `--resume`: Whether to resume training from the latest checkpoint.

##### Example Commands
If you get any error regarding ```"No module named 'x'"```, just manually run the command ```pip install -r requirements.txt``` or ```pip install -r requirements-jetson.txt``` inside the container.

###### Hyperparameter search and training
To perform hyperparameter tuning and training for different policies, use the following commands. Customize them with the arguments mentioned above:
```python
python3 training.py --policy ppo --tune

python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy dqn --checkpoint-dir dqn_checkpoints --tune
```

###### Directly Training
To train the policies without hyperparameter tuning remove the --tune argument:

```python
python3 training.py --policy ppo

python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy dqn --checkpoint-dir dqn_checkpoints
```

###### Resume training from checkpoint
To resume training from the latest checkpoint, you can use the --resume argument:
```
python3 training.py --policy ppo --resume
```
--checkpoint-dir checkpoints/ppo_policy

#### Test Trained Agents
To test the trained policies, you have the ```main.py```file:

```python
python3 main.py --policy ppo --checkpoint-dir checkpoints/ppo_policy
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
