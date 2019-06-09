#@author Bhoobhooz

import numpy as np
from random import sample
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


class kArmBandit:
    def __init__(self,k=10):
        self.action_values = np.random.normal(0,1,10)
        self.k = k

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

        for i in range(1,1001):
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


    def mab_greedy_epsilon(self,epsilon):

        ##choose the action which has max
        q_a = np.repeat(0.0, self.k)

        n_a = np.repeat(0.0, self.k)
        num_times_optimal_action_chosen = 0
        expected_reward_per_step = []
        expected_optimality_per_step = []

        total_reward = 0.0

        for i in range(1, 1001):

            chosen_action = np.argmax(q_a) if np.random.uniform(0,1,1)>epsilon else sample(range(0,10),1)
            reward = self.bandit(chosen_action)
            total_reward = total_reward + reward
            expected_reward_per_step.append(total_reward / i)
            is_optimal_action_chosen = 1 if chosen_action == np.argmax(self.action_values) else 0
            num_times_optimal_action_chosen = num_times_optimal_action_chosen + is_optimal_action_chosen
            expected_optimality_per_step.append(num_times_optimal_action_chosen / i)

            ##update q_a and n_a
            n_a[chosen_action] = n_a[chosen_action] + 1
            q_a[chosen_action] = q_a[chosen_action] + (1 / n_a[chosen_action]) * (reward - q_a[chosen_action])

        return expected_reward_per_step, expected_optimality_per_step, (total_reward / 1000)


if __name__ == '__main__':
    bandit_comparison_epsilon = []
    bandit_comparison_greedy = []
    bandit_comparison_epsilon_small =[]
    for i in range(1, 200):
        print(i)
        band_1 = kArmBandit()

        bandit_comparison_epsilon_iter = band_1.mab_greedy_epsilon(0.1)
        bandit_comparison_epsilon.append(bandit_comparison_epsilon_iter[0])

        bandit_comparison_epsilon_iter_small = band_1.mab_greedy_epsilon(0.01)
        bandit_comparison_epsilon_small.append(bandit_comparison_epsilon_iter_small[0])


        bandit_comparison_greedy_iter = band_1.mab_greedy()
        bandit_comparison_greedy.append(bandit_comparison_greedy_iter[0])

    rps_mean_ = pd.DataFrame(bandit_comparison_epsilon).mean(axis=0)
    rps_mean =pd.DataFrame(bandit_comparison_greedy).mean(axis=0)
    rps_mean_small_epsilon =pd.DataFrame(bandit_comparison_epsilon_small).mean(axis=0)




    # print(bandit_comparison[0])
        # mab_greedy_epsilon, mab_greedy =

    plt.plot(rps_mean_, 'r-',label = 'epsilon_0.1')
    plt.plot(rps_mean, 'b-',label = 'greedy')
    plt.plot(rps_mean_small_epsilon, 'g-',label = 'epsilon_0.01')

    plt.legend(loc = 'best')
    plt.savefig('mab_comparison.png')
    plt.show()
    # print(avg)
    # print(avg_)

