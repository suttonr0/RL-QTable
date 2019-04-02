import random
import gym
import tensorflow as tf
import numpy as np
import pandas as pd
import gym_notif  # Requires import even though IDE says it is unused
from timeit import default_timer
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from gym_notif.envs.mobile_notification import MobileNotification
from ml_metrics import MLMetrics


def get_q_state_encoding(possible_values: dict, notif: MobileNotification):
    # inputs
    # possible_values: dict values are a list of all possible values for their key category
    # (e.g. possible_states["time_of_day_states"] = ["morn", "afternoon", "evening"]

    # Q-State-Index is calculated of the combination of the indices of the three features in their possible value list
    output = np.zeros(possible_values["total_number_of_states"])
    q_state_index = 0
    q_state_index += possible_values["package_states"].index(notif.appPackage) * len(possible_values["category_states"]) * \
        len(possible_values["time_of_day_states"])  # List of package states
    q_state_index += possible_values["category_states"].index(notif.category) * len(possible_values["time_of_day_states"])  # List of package states
    q_state_index += possible_values["time_of_day_states"].index(notif.postedTimeOfDay)  # List of package states
    output[q_state_index] = 1
    return output


def split(a, n):
    k, m = divmod(len(a), n)  # Returns quotient and remainder for len(a) / n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))  # Returns a generator


# Agent is based off https://github.com/keon/deep-q-learning/blob/master/ddqn.py
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay queue
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    # Huber Loss function
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)  # if a <= delta, use (a^2 /2)
        quadratic_loss = clip_delta * (K.abs(error)) - 0.5 * K.square(clip_delta)  # a > delta, use del*|a| - (del^2 /2)
        l_delta = tf.where(cond, squared_loss, quadratic_loss)
        return K.mean(l_delta)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))  # Input Layer
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Linear activation for output
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training):
        if (np.random.rand() <= self.epsilon) and training:  #
            return random.randrange(self.action_size)
        # if not training:
        #    print(state)
        act_values = self.model.predict(state)  # Takes numpy array as input (or list of np arrays)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)  # Load model from file "name"

    def save(self, name):
        self.model.save_weights(name)  # Save the model to file "name"


if __name__ == "__main__":
    start_time = default_timer()

    # Cross Validation k value
    K_VALUE = 10
    k_fold_average_reward = 0

    # Create Environment (ensure this is outside of the cross-validation loop, otherwise the dataset will be randomly
    # shuffled between k values
    env = gym.make('notif-v0')
    env.render()

    action_size = env.action_space.n
    print("MAIN: Action size ", action_size)

    state_size = env.observation_space.n
    print("MAIN: State size", state_size)

    # Divide notification list into k equal parts
    k_parts_list = list(split(env.notification_list, K_VALUE))

    batch_size = 32

    end_time = default_timer()

    print("Setup time: {}".format(end_time - start_time))

    # For k in 10-fold cross validation
    for k_step in range(0, K_VALUE):
        # --- Create testing and training sets ---
        env.training_data = []
        env.testing_data = []

        # Create training data for all groups except the testing data group
        for group in k_parts_list:
            if group != k_parts_list[k_step]:
                env.training_data += group

        # Set testing data group
        env.testing_data = k_parts_list[k_step]

        start_time = default_timer()

        # Create the agent's parameters
        agent = DQNAgent(state_size, action_size)

        # Create the hyper parameters
        total_training_episodes = 10  # Was 50000, found 1000 to be good
        total_test_episodes = 100
        max_training_steps = len(env.training_data)  # Number of notifications per training episode
        max_testing_steps = len(env.testing_data)  # Number of notifications per testing episode

        env.training = True

        state = env.reset()

        for e in range(total_training_episodes):
            state = env.reset()
            state = get_q_state_encoding(env.info, state)
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            for step in range(max_training_steps):
                # env.render()
                action = agent.act(state, env.training)
                next_state, reward, done, _ = env.step(bool(action))
                total_reward += reward
                next_state = get_q_state_encoding(env.info, next_state)
                # reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    agent.update_target_model()
                    print("TRAINING: episode: {}/{}, total reward: {}, steps: {}, e: {:.2}"
                          .format(e, total_training_episodes, total_reward, step, agent.epsilon))
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

        end_time = default_timer()

        print("Training time: {}".format(end_time - start_time))

        # ----- Using the Trained DQN (Testing) -----
        start_time = default_timer()
        env.training = False
        env.reset()
        list_tot_rewards = []

        for e in range(total_test_episodes):
            state = env.reset()
            done = False
            state = get_q_state_encoding(env.info, state)
            state = np.reshape(state, [1, state_size])
            print("Testing Episode {}".format(e))
            total_reward = 0
            metr = MLMetrics()
            for step in range(max_testing_steps):
                # env.render()
                action = agent.act(state, env.training)
                # print("Testing phase action: {}".format(action))
                next_state, reward, done, _ = env.step(bool(action))
                metr.update(bool(action), bool(action) == bool(reward))
                total_reward += reward
                next_state = get_q_state_encoding(env.info, next_state)
                next_state = np.reshape(next_state, [1, state_size])
                state = next_state
                if done:
                    print("TESTING: episode: {}/{}, total reward: {}, steps: {}, e: {:.2}"
                          .format(e, total_test_episodes, total_reward, step, agent.epsilon))
                    list_tot_rewards.append(total_reward)
                    print("TP {}, TN {}, FP {}, FN {}, Prec {}, Rec {}, F1 {}, Acc {}".format(
                        metr.true_pos, metr.true_neg, metr.false_pos, metr.false_neg, metr.calc_precision(),
                        metr.calc_recall(), metr.calc_f1_score(), metr.calc_accuracy()))
                    print("Click-through Rate {}".format(metr.calc_click_through_rate()))
                    break
        print("Score over time: {} for k iteration {}".format(sum(list_tot_rewards) / total_test_episodes, k_step))
        k_fold_average_reward += (sum(list_tot_rewards) / total_test_episodes)
        end_time = default_timer()
        print("Testing time: {}".format(end_time - start_time))

    env.close()
    print("Final average reward across all {} validations: {}".format(K_VALUE, k_fold_average_reward/K_VALUE))
