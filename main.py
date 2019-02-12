import gym
import gym_notif
import csv
from notification import Notification

# Create Environment
env = gym.make('notifenv-v0')

action = "JUMP"
new_state, reward, done, info = env.step(action)
print(reward)
action = "JUMP"
new_state, reward, done, info = env.step(action)
print(reward)
action = "CROUCH"
new_state, reward, done, info = env.step(action)
print(reward)

# notification_list = []
# # Takes columns in order as they are presented in csv
# fieldnames = ("index", "action", "appPackage", "category", "postedTimeOfDay")
#
# with open("notif_user_2.csv") as csvfile:
#     reader = csv.DictReader(csvfile, fieldnames)
#     for row in reader:
#         notification_list.append(Notification(row["appPackage"], row["category"], row["postedTimeOfDay"]))
#
# notification_list.pop(0)  # Remove Notif object containing table headers
#
# # Find all possible values for packages, categories and ToD
# package_states = []
# category_states = []
# time_of_day_states = []
# for item in notification_list:
#     if item.appPackage not in package_states:
#         package_states.append(item.appPackage)
#     if item.category not in category_states:
#             category_states.append(item.category)
#     if item.postedTimeOfDay not in time_of_day_states:
#             time_of_day_states.append(item.postedTimeOfDay)
#
# print(package_states)
# print(category_states)
# print(time_of_day_states)
#
# total_num_states = len(package_states) * len(category_states) * len(time_of_day_states)
# print("Total number of Notification states: " + str(total_num_states))

