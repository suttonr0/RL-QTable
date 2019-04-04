import random
import csv
import gym
import numpy as np
import gym_notif  # Requires import even though IDE says it is unused
from timeit import default_timer
from gym_notif.envs.mobile_notification import MobileNotification
from ml_metrics import MLMetrics, OverallMetrics


def get_q_state_index(states: dict, notif: MobileNotification, num_features: int):
    # inputs
    # states: dict values are a list of all possible values for their key category
    # (e.g. states["time_of_day_states"] = ["morn", "afternoon", "evening"]

    # Q-State-Index is calculated of the combination of the indices of the features in their possible value list
    q_state_index = 0
    if num_features >= 4:
        q_state_index += states["day_of_week_states"].index(notif.postedDayOfWeek) * len(states["time_of_day_states"]) * \
                         len(states["category_states"]) * len(states["package_states"])
    if num_features >= 3:
        q_state_index += states["time_of_day_states"].index(notif.postedTimeOfDay) * \
                         len(states["category_states"]) * len(states["package_states"])  # List of package states
    if num_features >= 2:
        q_state_index += states["category_states"].index(notif.category) * len(states["package_states"])
    q_state_index += states["package_states"].index(notif.appPackage)  # List of package states
    return q_state_index


def split(a, n):
    k, m = divmod(len(a), n)  # Returns quotient and remainder for len(a) / n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))  # Returns a generator


if __name__ == "__main__":
    start_time = default_timer()

    # Cross Validation k value
    K_VALUE = 10
    k_fold_average_reward = 0
    k_metrics = []
    training_metrics = []

    # Create Environment (ensure this is outside of the cross-validation loop, otherwise the dataset will be randomly
    # shuffled between k values
    env = gym.make('notif-v0')
    env.render()

    num_features = env.feat_number
    print("MAIN: NUMBER OF FEATURES ", num_features)

    action_size = env.action_space.n
    print("MAIN: Action size ", action_size)

    state_size = env.observation_space.n
    print("MAIN: State size", state_size)

    # Divide notification list into 10 equal parts
    k_parts_list = list(split(env.notification_list, K_VALUE))

    end_time = default_timer()

    print("Setup time: {}".format(end_time - start_time))

    # For k in 10-fold cross validation
    for k_step in range(0, K_VALUE):
        env.training_data = []
        env.testing_data = []

        # Create training data for all groups except the testing data group
        for group in k_parts_list:
            if group != k_parts_list[k_step]:
                env.training_data += group

        # Set testing data group
        env.testing_data = k_parts_list[k_step]

        start_time = default_timer()

        # Create Q-Table
        qtable = np.zeros((state_size, action_size))

        # Create the hyper parameters
        total_training_episodes = 1000  # Was 50000
        total_test_episodes = 100
        max_training_steps = len(env.training_data)  # Number of notifications per training episode
        max_testing_steps = len(env.testing_data)  # Number of notifications per testing episode

        learning_rate = 0.7  # Was 0.7
        gamma = 0.618  # Discount rate

        # Exploration parameters
        epsilon = 1.0  # Exploration rate
        max_epsilon = 1.0  # Exploration probability at the start
        min_epsilon = 0.01  # Min exploration probability
        decay_rate = 0.005  # Exponential decay rate for exploration, was 0.01

        env.training = True

        # ----- The Q-Learning Algorithm -----
        print("Training...")

        for episode in range(total_training_episodes):
            # Reset the environment
            state = env.reset()
            done = False
            total_reward = 0
            # Each step changes the state to another notification
            for step in range(max_training_steps):
                # Get random number for exploration/exploitation
                exp_exp_tradeoff = random.uniform(0, 1)

                # If this random number > epsilon --> exploitation (take largest Q value from the Q-table)
                if exp_exp_tradeoff > epsilon:
                    action = np.argmax(qtable[get_q_state_index(env.info, state, num_features), :])

                # Else do a random choice --> exploration
                else:
                    action = env.action_space.sample()

                # Take the action (a) and observe the outcome state (s') and reward (r)
                new_state, reward, done, info = env.step(bool(action))
                total_reward += reward

                # Update Q(s,a) using the Bellman equation
                qtable[get_q_state_index(env.info, state, num_features), action] = qtable[get_q_state_index(env.info, state, num_features), action] + \
                    learning_rate * (reward + gamma * np.max(qtable[get_q_state_index(env.info, new_state, num_features), :]) - qtable[get_q_state_index(env.info, state, num_features), action])

                # Update to the new state
                state = new_state

                # If done (i.e. passed through all states in the training set) then finish episode
                if done:
                    print("TRAINING: k:{}, episode: {}/{}, total reward: {}, steps: {}, epsilon: {}"
                          .format(k_step, episode, total_training_episodes, total_reward, step, epsilon))
                    if k_step == 0:
                        training_metrics.append([episode, total_reward/step, epsilon])
                    break

            episode += 1
            # Reduce epsilon (to reduce exploration over time)
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

        end_time = default_timer()

        training_time = end_time - start_time
        print("Training time: {}".format(training_time))
        # print(qtable)

        # ----- Using the Trained Q-Table -----
        start_time = default_timer()
        env.training = False
        env.reset()
        list_tot_rewards = []
        metric_list = OverallMetrics()

        for episode in range(total_test_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            metr = MLMetrics()
            print("EPISODE ", episode)
            for step in range(max_testing_steps):
                # env.render
                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(qtable[get_q_state_index(env.info, state, num_features), :])
                new_state, reward, done, info = env.step(bool(action))
                # Actual action equals X-NOR(predicted, reward)
                metr.update(bool(action), not(bool(action) != bool(reward)))
                total_reward += reward
                if done:
                    metric_list.update(metr)
                    print(metr)
                    break
                state = new_state

        end_time = default_timer()
        testing_time = end_time - start_time
        print("Testing time: {}".format(testing_time))
        print("Average accuracy: {}".format(metric_list.average_accuracy()))
        k_metrics.append(metric_list.get_average_metrics(k_step) + [training_time, testing_time])

    csv_name = env.CSV_FILE.split('/')[1].split('.')[0]  # Removes directory and file extension from the env's CSV name
    env.close()

    # ----- Write Average ML metrics for each k-step to csv -----
    file_1 = open("csv_output/feat{}_QTable.csv".format(num_features), "w", newline='')  # Newline override to prevent blank rows in Windows
    writer = csv.writer(file_1)
    writer.writerow(["k_value", "Precision", "Accuracy", "Recall", "F1 Score", "Click_Through", "Train time", "Test time"])
    for row in k_metrics:
        writer.writerow(row)
    file_1.close()

    # ----- Write reward and epsilon values across episodes to csv -----
    file_1 = open("csv_output/feat{}_k0traindata_QTable.csv".format(num_features), "w", newline='')
    writer = csv.writer(file_1)
    writer.writerow(["Episode", "Percentage Reward", "Epsilon"])
    for row in training_metrics:
        writer.writerow(row)
    file_1.close()
