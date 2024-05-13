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
python3 -m venv .venv
source .venv/bin/activate
pip install numpy gymnasium pettingzoo matplotlib pandas ray "ray[tune]" tree typer scikit-image optuna torch lz4 tensorflow supersuit

```
### Setup on Windows
```
python -m venv .venv
.venv\Scripts\activate
pip install numpy gymnasium pettingzoo matplotlib pandas ray "ray[tune]" tree typer scikit-image optuna torch lz4 tensorflow supersuit 

```
When done, you can run the following command to exit the virtual environment.
```
deactivate
```

## Usage
Monte-Carlo simulation. No trained agent involved. Edit simulation parameters at the end of the file ```gym_env.py``` .
### Usage on Mac or Linux
```
python3 gym_env.py
```
### Usage on Windows
```
python gym_env.py 
```

Acutal Trained Agent. Training for PPO (working on DQN as well) 
### Usage on Mac or Linux
```
python3 train.py
```
### Usage on Windows
```
python train.py
```
