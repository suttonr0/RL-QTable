import random
import gym
import numpy as np
import pandas as pd
import gym_notif  # Requires import even though IDE says it is unused
from gym_notif.envs.mobile_notification import MobileNotification


def get_q_state_index(oneHotList: list, notif: MobileNotification):
    # Takes a list of one-hot encodings and a notification object and calculates the q-table index
    qStateIndex = 0
    for oneHotEncoding in oneHotList:  # For each category
        for key in oneHotEncoding:  # For each key in that category's one hot encoding
            if key == notif.appPackage:
                test = oneHotEncoding[key].tolist()
                print(test.index(1))
                #len(one_hot_list[item])
            if key == notif.category:
                test = oneHotEncoding[key].tolist()
                print(test.index(1))
                #len(one_hot_list[item])
            if key == notif.postedTimeOfDay:
                test = oneHotEncoding[key].tolist()
                print(test.index(1))
                #len(one_hot_list[item])


# Create Environment
env = gym.make('notif-v0')
env.render()

action_size = env.action_space
print("MAIN: Action size ", action_size)

state_size = env.observation_space
print("MAIN: State size", state_size)

# Get One-Hot encodings for each category's values
package_one_hot = pd.Series(env.info['package_states']).str.get_dummies(', ')
category_one_hot = pd.Series(env.info['category_states']).str.get_dummies(', ')
tod_one_hot = pd.Series(env.info['time_of_day_states']).str.get_dummies(', ')

one_hot_list = []
one_hot_list.append(package_one_hot)
one_hot_list.append(category_one_hot)
one_hot_list.append(tod_one_hot)

# pd.Series(time_of_day_states).str.get_dummies(', ') is of type pandas.core.frame.DataFrame
# Can be indexed by pd.Series(...)...["afternoon"], returning a
# pandas.core.series.Series with its one-hot encoding. This encoding can then be indexed as a normal array

get_q_state_index(one_hot_list, env.state)
get_q_state_index(one_hot_list, MobileNotification(True, "com.instagram.android", "unknown", "morn"))

# Following Code is from https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Taxi-v2/Q%20Learning%20with%20OpenAI%20Taxi-v2%20video%20version.ipynb

# print(action_size.sample())

# Create Q-Table
qtable = np.zeros((81, 2))

# Create the hyper parameters
total_episodes = 10000  # Was 50000
total_test_episodes = 100
max_steps = 99  # Max steps per episode

learning_rate = 0.7
gamma = 0.618  # Discount rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at the start
min_epsilon = 0.01  # Min exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration

# TODO: ALL GOOD ABOVE HERE

# # The Q-Learning Algorithm
# for episode in range(total_episodes):
#     # Reset the environment
#     state = env.reset()
#     step = 0
#     done = False
#     print(episode)
#     for step in range(max_steps):
#         # Choose an action a in the current world state (s)
#         # First, randomize a number
#         exp_exp_tradeoff = random.uniform(0, 1)
#
#         # If this random number > epsilon --> exploitation (take largest Q value)
#         if exp_exp_tradeoff > epsilon:
#             action = np.argmax(qtable[state, :])
#
#         # Else do a random choice --> exploration
#         else:
#             action = env.action_space.sample()
#
#         # Take the action (a) and observe the outcome state (s') and reward (r)
#         new_state, reward, done, info = env.step(action)
#
#         # Update Q(s,a) using the Bellman equation
#         qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma *
#                                                                          np.max(qtable[new_state, :]) - qtable[state, action])
#
#         # Update to the new state
#         state = new_state
#
#         # If done then finish episode
#         if done:
#             break
#
#     # # Move to next episode
#     # episode += 1
#
#     # Reduce epsilon (to reduce exploration over time)
#     epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
#
# print("OK")
# print(qtable)
#
# # Now after the Q-table is trained, it can be used for the application
# env.reset()
# rewards = []
#
# for episode in range(total_test_episodes):
#     state = env.reset()
#     step = 0
#     done = False
#     total_rewards = 0
#     # print("****************************************************")
#     # print("EPISODE ", episode)
#
#     for step in range(max_steps):
#         # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
#         env.render()
#         # Take the action (index) that have the maximum expected future reward given that state
#         action = np.argmax(qtable[state, :])
#
#         new_state, reward, done, info = env.step(action)
#
#         total_rewards += reward
#
#         if done:
#             rewards.append(total_rewards)
#             # print ("Score", total_rewards)
#             break
#         state = new_state
# env.close()
# print("Score over time: " + str(sum(rewards) / total_test_episodes))

