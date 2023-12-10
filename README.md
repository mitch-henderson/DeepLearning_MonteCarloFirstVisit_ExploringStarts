# DeepLearning MonteCarloFirstVisit ExploringStarts

![DeepLearning_MonteCarloFirstVisit_ExploringStarts](https://github.com/mitch-henderson/DeepLearning_MonteCarloFirstVisit_ExploringStarts/blob/main/mitch___h_Blackjack_Monte_Carlo_Simulation._Miami_vice._downtow_fc409bf4-e1eb-46f2-b70a-7a0030be15f5.png)


I am applying the deep learning models Monte Carlo First Visit-Prediction, 
and Monte Carlo Exploring Starts (ES) [a type of on-policy control](https://towardsdatascience.com/on-policy-v-s-off-policy-learning-75089916bc2f), to the game of Blackjack.


# Blackjack Monte Carlo Simulation

This repo contains a Monte Carlo simulation for the game of Blackjack using the `gym` environment.

## Dependencies

- `gym`
- `numpy`
- `matplotlib`
- `seaborn`
- `tqdm`
- `pathlib`
- `pickle`

## Overview

The main goal of this simulation is to determine the best possible action (either to hit or stick) based on the current hand of the player, the visible card of the dealer, and whether or not the player has a usable ace. 

The repository contains functions to:
- Play a single game of blackjack.
- Define dealer and player policies.
- Execute Monte Carlo on-policy.
- Run Monte Carlo with exploring starts.
- Visualization of the results.

## Primary Algorithm Toolset

The code in this repository primarily leverages the following algorithms and tools:

- **Monte Carlo Method**: Used for estimating the value of states in the Blackjack environment. This algorithm generates samples from the state space to estimate state values.
  
- **OpenAI's `gym` Library**: This is a toolkit for developing and comparing reinforcement learning algorithms. In our code, we use the `BlackjackEnv` from the toy text environments provided by `gym`.

- **Seaborn and Matplotlib**: These Python data visualization libraries are used for visualizing the results of the Monte Carlo simulations.

- **Numpy**: This library supports large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

To get started with the toolset, ensure you have all the dependencies installed. You can generally install them using pip:

```bash
pip install gym numpy seaborn matplotlib
```


## How to Use

1. Set up the environment:

```python
import gym.envs.toy_text.blackjack as bj
env = bj.BlackjackEnv()
```

## Reset the environment:

```python
env.reset()
```

## Sample action and observation spaces:
```python
env.action_space.sample()
env.observation_space[0].n
env.observation_space[1].n
env.observation_space[2].n
```

## Play a single game:
```python
env.seed(42)
print('Initial state:', env.reset())
print('Playing one game...')
play(env, player_policy)
```
## Execute Monte Carlo simulations:
```python
run_monte_carlo_on_policy()
run_monte_with_exploring_starts(num_episodes_es)
```

## Visualize the results:
```python
plot_monte_carlo_on_policy(states, titles)
plot_monte_carlo_with_exploring_starts(policy_values, titles)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
