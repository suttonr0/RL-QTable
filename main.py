import random
import gym
import numpy as np
import pandas as pd
import gym_notif  # Requires import even though IDE says it is unused
from gym_notif.envs.mobile_notification import MobileNotification


def get_q_state_index(possible_values: dict, notif: MobileNotification):
    # inputs
    # possible_values: dict values are a list of all possible values for their key category
    # (e.g. possible_states["time_of_day_states"] = ["morn", "afternoon", "evening"]

    # Q-State-Index is calculated of the combination of the indices of the three features in their possible value list
    q_state_index = 0
    q_state_index += possible_values["package_states"].index(notif.appPackage) * len(possible_values["category_states"]) * \
        len(possible_values["time_of_day_states"])  # List of package states
    q_state_index += possible_values["category_states"].index(notif.category) * len(possible_values["time_of_day_states"]) # List of package states
    q_state_index += possible_values["time_of_day_states"].index(notif.postedTimeOfDay) # List of package states

    return q_state_index


# Create Environment
env = gym.make('notif-v0')
env.render()

action_size = env.action_space.n
print("MAIN: Action size ", action_size)

state_size = env.observation_space.n
print("MAIN: State size", state_size)

# Sample code for using get_q_state_index()
counter = np.zeros(81)
for i in range(0, 628):
    print("Notif Number" + str(i))
    print(get_q_state_index(env.info, env.state))
    counter[get_q_state_index(env.info, env.state)] += 1
    env.step(False)
print(counter)
# print("OVERALL: " + str(counter.sum()))

# Following Code is from https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Taxi-v2/Q%20Learning%20with%20OpenAI%20Taxi-v2%20video%20version.ipynb

# print(action_size.sample())

# Create Q-Table
qtable = np.zeros((state_size, action_size))
print(qtable)

# Create the hyper parameters
total_episodes = 3000  # Was 50000
total_test_episodes = 100
max_steps = 99  # Max steps per episode

learning_rate = 0.7
gamma = 0.618  # Discount rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at the start
min_epsilon = 0.01  # Min exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration

# TODO: Fix Q-Learning Algorithm and Implementation

# The Q-Learning Algorithm
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    print(episode)
    for step in range(max_steps):
        # Choose an action a in the current world state (s)
        # First, randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        # If this random number > epsilon --> exploitation (take largest Q value)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[get_q_state_index(env.info, state), :])

        # Else do a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state (s') and reward (r)
        new_state, reward, done, info = env.step(bool(action))

        # Update Q(s,a) using the Bellman equation
        qtable[get_q_state_index(env.info, state), action] = qtable[get_q_state_index(env.info, state), action] + \
            learning_rate * (reward + gamma * np.max(qtable[get_q_state_index(env.info, new_state), :]) - qtable[get_q_state_index(env.info, state), action])

        # Update to the new state
        state = new_state

        # If done then finish episode
        if done:
            break

    episode += 1

    # Reduce epsilon (to reduce exploration over time)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

print("OK")
print(qtable)

# Now after the Q-table is trained, it can be used for the application
env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    # print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
        # env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[get_q_state_index(env.info, state), :])

        new_state, reward, done, info = env.step(bool(action))

        total_rewards += reward
        if done:
            rewards.append(total_rewards)
            print("Score", total_rewards)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards) / total_test_episodes))

