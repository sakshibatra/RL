#@author Bhoobhooz

import numpy as np
from random import sample
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


class GradientBandit:
    def __init__(self,k=10,true_mean=4):
        self.action_values = np.random.normal(true_mean,1,10)
        self.k = k


    def bandit(self,chosenAction):
        return np.random.normal(self.action_values[chosenAction],1,1)[0]

    def compute_prob(self,chosenAction,prob_pref):
        return np.exp(prob_pref[chosenAction])/np.sum()

    def gradient_bandit(self,alpha):
        ##choose the action which has max
        h_a = np.repeat(0.0,self.k)

        num_times_optimal_action_chosen =0
        expected_reward_per_step = []
        expected_optimality_per_step = []

        total_reward = 0.0

        for i in range(1,1001):
            chosen_action = np.argmax(h_a)
            reward = self.bandit(chosen_action)
            total_reward = total_reward+reward
            reward_baseline = total_reward/i
            expected_reward_per_step.append(reward_baseline)

            is_optimal_action_chosen = 1 if chosen_action == np.argmax(self.action_values) else 0
            num_times_optimal_action_chosen = num_times_optimal_action_chosen +is_optimal_action_chosen
            expected_optimality_per_step.append(num_times_optimal_action_chosen/i)

            ##update h_a and p_a
            h_a[chosen_action] = h_a[chosen_action]+alpha * (reward-reward_baseline)*(1-compute_prob(chosen_action,h_a))


        return expected_reward_per_step,expected_optimality_per_step,(total_reward/1000)


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

