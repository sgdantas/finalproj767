# finalproj767

## Final project for COMP-767 at McGill University
## Group: Joao Pedro de Carvalho (McGill ID 260642102), Peyman Kafaei (McGill ID 260780776), Stefano Giacomazzi Dantas(McGill ID 260642029)
## Goal

### Study the learning behaviour of <br/> [Learning Heuristics for the TSP by Policy Gradient](https://link.springer.com/chapter/10.1007%2F978-3-319-93031-2_12) in terms of episodes and hyperparameter values.

### Investigate the effects of removing the baseline policy from the objective function, and assess the impact on learning.

### Implement TD learning instead of the default Monte Carlo sampling in REINFORCE.

### Implement eligibility traces and investigate its performance.


## Requirements

- [Python 3.5+](https://anaconda.org/anaconda/python)
- [TensorFlow 1.3.0+](https://www.tensorflow.org/install/)
- [Tqdm](https://pypi.python.org/pypi/tqdm)

## Usage


The code was adapted from https://github.com/MichelDeudon/encode-attend-navigate

The parameters are located at the beginning of the code for the .py files or in the Config block for .ipynb

The main parameters are:

### Data
- '--batch_size' : the batch size for training
- '--max_length' : number of cities for training

### Model

- It is possible to change the NN parameters (number of attention heads, number of neurons, etc). However, the performance may vary compared to the results reported

### Train / test parameters
- '--nb_steps' : number of epochs for training
- '--lr_start': actor learning rate
- '--lr_decay_rate' : learning rate decay rate
- '--temperature' : temperature for the policy distribution
- '--C' : clipping parameter


- To train a model just run blocks 1.DataGenerator, 2.Config, 3.Model and 4.Train with the Jupyter Notebook (for files .ipynb). If the file is .py, just execute the code

### Files Specification


- K1_Neural_Reinforce.ipynb	: Model with K = 1
- K5_Neural_Reinforce.ipynb	: Model with K = 5
- Neural_Reinforce-NoCritic.ipynb	: Model without baseline
- Neural_Reinforce.ipynb* : base model
- PlotMemory.ipynb	: code used to plot the memory learning curves
- n_step_return.py	: n step TD, eligibility traces implementation
- data_generator.py*	: generates the trajectory
- graph.py	: code used to generate plots
- utils.py* : utility functions, 

\* original or adapted from https://github.com/MichelDeudon/encode-attend-navigate
