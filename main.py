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

# Create Q-Table
qtable = np.zeros((state_size, action_size))
# print(qtable)

# Create the hyper parameters
total_episodes = 100  # Was 50000
total_test_episodes = 100
max_steps = env.info['number_of_notifications']  # Max steps per episode. Should be the number of notifications under investigation for this episode

learning_rate = 0.7  # Was 0.7
gamma = 0.618  # Discount rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at the start
min_epsilon = 0.01  # Min exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration, was 0.01

# ----- The Q-Learning Algorithm -----
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    done = False
    print("Training Episode: {}".format(episode))
    # Each step changes the state to a new notification
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
            # print("SAMPLE ACTION: " + str(action))

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

print(qtable.transpose())

# ----- Using the Trained Q-Table -----
env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
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
            rewards.append(total_rewards)  # Division by step can be added to get percentage
            print("Score", total_rewards)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards) / total_test_episodes))

