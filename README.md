# DeepLearning MonteCarloFirstVisit ExploringStarts

![DeepLearning_MonteCarloFirstVisit_ExploringStarts](https://github.com/mitch-henderson/DeepLearning_MonteCarloFirstVisit_ExploringStarts_CSPB3202/blob/main/2023_08_mitch___h_deep_learning_models_Monte_Carlo_First_Visit-Predicti.png)


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
