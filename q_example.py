import numpy as np
import gym
import random

# Create environment
env = gym.make("Taxi-v2")
env.render()

action_size = env.action_space.n
print("Action size ", action_size)

state_size = env.observation_space.n
print("State size", state_size)

# Create Q-Table
qtable = np.zeros((state_size, action_size))
print(qtable)

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
            action = np.argmax(qtable[state, :])

        # Else do a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state (s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a) using the Bellman equation
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma *
                                                                         np.max(qtable[new_state, :]) - qtable[state, action])

        # Update to the new state
        state = new_state

        # If done then finish episode
        if done:
            break

    # # Move to next episode
    # episode += 1

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
    # print("EPISODE ", episode)

    for step in range(max_steps):
        # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
        env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            # print ("Score", total_rewards)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards) / total_test_episodes))
