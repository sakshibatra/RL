#@author Bhoobhooz

import random
import numpy as np
from random import sample
import matplotlib
import matplotlib.pyplot as plt


class kArmBandit:
    def __init__(self):
        self.action_values = np.random.normal(0,1,10)
        self.k = 10
        self.epsilon = 0.1

    def bandit(self,chosenAction):
        return np.random.normal(self.action_values[chosenAction],1,1)[0]

    def mab_greedy(self):
        ##choose the action which has max
        q_a = np.repeat(0.0,self.k)

        n_a = np.repeat(0.0,self.k)
        num_times_optimal_action_chosen =0
        expected_reward_per_step = []
        expected_optimality_per_step = []

        total_reward = 0.0

        for i in range(1,1000):
            chosen_action = np.argmax(q_a)
            reward = self.bandit(chosen_action)
            total_reward = total_reward+reward
            expected_reward_per_step.append(total_reward/i)
            is_optimal_action_chosen = 1 if chosen_action == np.argmax(self.action_values) else 0
            num_times_optimal_action_chosen = num_times_optimal_action_chosen +is_optimal_action_chosen
            expected_optimality_per_step.append(num_times_optimal_action_chosen/i)

            ##update q_a and n_a
            n_a[chosen_action] = n_a[chosen_action] +1
            q_a[chosen_action] = q_a[chosen_action] + (1/n_a[chosen_action])*(reward - q_a[chosen_action])


        return expected_reward_per_step,expected_optimality_per_step,(total_reward/1000)


    def mab_greedy_epsilon(self):
        ##choose the action which has max
        q_a = np.repeat(0.0, self.k)

        n_a = np.repeat(0.0, self.k)
        num_times_optimal_action_chosen = 0
        expected_reward_per_step = []
        expected_optimality_per_step = []

        total_reward = 0.0

        for i in range(1, 1000):

            chosen_action = np.argmax(q_a) if np.random.uniform(0,1,1)>self.epsilon else sample(range(0,10),1)
            reward = self.bandit(chosen_action)
            total_reward = total_reward + reward
            expected_reward_per_step.append(total_reward / i)
            is_optimal_action_chosen = 1 if chosen_action == np.argmax(self.action_values) else 0
            num_times_optimal_action_chosen = num_times_optimal_action_chosen + is_optimal_action_chosen
            expected_optimality_per_step.append(num_times_optimal_action_chosen / i)

            ##update q_a and n_a
            n_a[chosen_action] = n_a[chosen_action] + 1
            q_a[chosen_action] = q_a[chosen_action] + (1 / n_a[chosen_action]) * (reward - q_a[chosen_action])
        print(expected_reward_per_step)
        return expected_reward_per_step, expected_optimality_per_step, (total_reward / 1000)


if __name__ == '__main__':
    band_1 = kArmBandit()
    print(band_1.action_values)
    rps,ops,avg = band_1.mab_greedy()
    rps_,ops_,avg_ = band_1.mab_greedy_epsilon()
    print(rps)
    print(rps_)

    plt.plot(rps,'r-')
    plt.plot(rps_,'b*-')

    plt.show()
    print(avg)
    print(avg_)

