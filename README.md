# DQN for Lunar Lander

DQN for Lunar Lander

## Getting Started

This the repository for application of DQN to Lunar Lander.

### Prerequisites

What things you need to install the software

```
Gym, PyTorch, Numpy, matplotlib
```


## Running the process
AgentRunner is where you can run the whole process
To run learning for single set of hyperparameters, uncomment below under main and run main

```
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
alpha = 0.0005
target_update = 20
max_iter = 2000
batch_size = 64
tau = 0.001
dropout_ratio = 0

agent = Agent(gamma, epsilon_start, epsilon_end, epsilon_decay, alpha, target_update, max_iter, tau, batch_size, dropout_ratio)
agent.train()
agent.load_model()
agent.test()
```
To run for hyperparameter tuning, uncomment below and run main
```
tuner = HyperparameterTuner()
tuner.gamma_analysis()
tuner.alpha_analysis()
tuner.epsilon_decay_analysis()
tuner.target_update_analysis()
```

#
## Authors

* **Sizhang Zhao** - *Initial work* 


