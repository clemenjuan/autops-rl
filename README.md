# Master Thesis



## Getting started

ToDo [Use the template at the bottom](#editing-this-readme)!


## Setup
In the folder where you want to create the directory:

```
git clone https://gitlab.lrz.de/clemente.juan/masterthesis_git.git
cd masterthesis_git
```
It is strongly recommended that you use a virtual environment in order to manage dependencies and keep your projects organized without interfering with other projects or the global Python installation. You can create one by:
### Setup on Mac or Linux
```
# Creates the environment
python3 -m venv .venv

# Starts the environment
source .venv/Scripts/activate

# Installs all necessary packages
pip install numpy gymnasium pettingzoo matplotlib pandas ray "ray[tune]" tree typer scikit-image optuna torch lz4 tensorflow supersuit
```

### Setup on Windows
```
# Creates the environment
python -m venv .venv

# Starts the environment
.venv\Scripts\activate

# Installs all necessary packages
pip install numpy gymnasium pettingzoo matplotlib pandas ray "ray[tune]" tree typer scikit-image optuna torch lz4 tensorflow supersuit 

```
When done, you can run the following command to exit the virtual environment.
```
deactivate
```

## Usage
## Monte-Carlo simulation. 
No trained agent involved. Edit simulation parameters at the end of the file ```FSS_env.py``` .
### Usage on Mac or Linux
```
python3 FSS_env.py
```
### Usage on Windows
```
python FSS_env.py 
```

## Train Agents
Training for different policies (PPO, DQN, A2C, A3C, IMPALA).
### Usage
Edit the common configuration setup function in training.py to include gpu resources or add more parallelism to the training process. 
To perform hyperparameter tuning and training for different policies, run the following commands (```python``` only for Windows):
```
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy ppo --checkpoint-dir ppo_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy dqn --checkpoint-dir dqn_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a2c --checkpoint-dir a2c_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a3c --checkpoint-dir a3c_checkpoints --tune
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy impala --checkpoint-dir impala_checkpoints --tune
````
If you only want to train (or keep training from last checkpoint) the policies without tuning, omit the --tune argument:
```
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy ppo --checkpoint-dir ppo_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy dqn --checkpoint-dir dqn_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a2c --checkpoint-dir a2c_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy a3c --checkpoint-dir a3c_checkpoints
python3 training.py --framework torch --stop-iters 20 --stop-reward 500000 --policy impala --checkpoint-dir impala_checkpoints
```
