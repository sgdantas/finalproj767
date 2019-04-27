# finalproj767

## Final project for COMP-767 at McGill University
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


- To train a model from scratch (data is generated on the fly), run blocks 1.DataGenerator, 2.Config, 3.Model and 4.Train with the Jupyter Notebook (Neural_Reinforce.ipynb). You can change parameters in the Config block. Default parameters should replicate results reported in our paper (2D TSP50).

- If training is successful, the model will be saved in a "save" folder (filename depends on config) and training statistics will be reported in a "summary" folder. To visualize training on tensorboard, run:
```
> tensorboard --logdir=summary
```

- To test a trained model, run block 5.Test with the Jupyter Notebook (Neural_Reinforce.ipynb).
