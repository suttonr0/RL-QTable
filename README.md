# Mobile Notification Reinforcement Learning System
This project is intended for use with the gym-notif environment found [here](https://github.com/suttonr0/gym-notif).
Input CSV files containing notification data should be placed within a `csv_files` folder in this project, and should be pointed to by the gym-notif project.
A sample CSV file is shown in `sampleInput.csv`.

A Q-Learning system is available in `q_table.py`, and a Deep Q-Learning system is available in `deep_q_learning.py`.

# Reinforcement Learning System Components
## Agent
The agent is the Q-table / Deep Q-Network Reinforcement Learning algorithm acting on the environment.

## States
The states for the system are the possible permutations of notification features for the available mobile notifications themselves. 

## Action
The actions are to show or hide the notification from the user.

## Environment
The environment is a user interacting with a mobile device as simulated by the `gym-notif` Gym environment.

## Rewards
The reward is feedback to the environment. In the case of synthetic notification data, a ground truth of the correct action 
is generated and if the system prediction matches this ground truth, a positive reward is generated. 