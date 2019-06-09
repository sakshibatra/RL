import numpy as np
import matplotlib.pyplot as plt
import math

class RandomWalk:
    def __init__(self):
        self.state = 'C'
        self.action_space = ['left','right']
        self.RIGHT_TERMINAL_STATE ='F'
        self.LEFT_TERMINAL_STATE = '@'

    def action(self,state):
        action_chosen = 'left' if np.random.rand()<0.5 else 'right'
        return(action_chosen)


    def reward(self,state,action):
        action_val = -1 if action=='left' else 1
        next_state = chr(ord(state)+action_val)
        reward = 1 if next_state==self.RIGHT_TERMINAL_STATE else 0
        done = True if next_state ==self.RIGHT_TERMINAL_STATE or next_state==self.LEFT_TERMINAL_STATE else False
        return (state,action,next_state,reward,done)


##initialize value_state mapping to 0
value={'@':0,'A':0.5,'B':0.5,'C':0.5,'D':0.5,'E':0.5,'F':0}
gamma = 1
def sample_episode(env,alpha=0.1):
    ##Values(state) = value(state)+alpha[rewatd+lambda*
    """

    :param reward:
    :param state:
    :param next_state:
    :return: value_estimate
    """
    ##for each step of episode update value
    state = env.state
    while True:
        action = env.action(state)

        state, action, next_state, reward, done = env.reward(state,action)
        #print(env.reward(state,action))
        if state in ('A','B','C','D','E'):
            value[state] = value.get(state)+alpha *(reward+(gamma * value.get(next_state))-value.get(state))
        if done:
            break;
        state = next_state


def rms(predicted_values,true_values):
    """

    :return: root mean squared error
    """

    error = np.array(true_values) - np.array(predicted_values)
    return np.sqrt(np.mean(error ** 2))

if __name__== '__main__':
    randomwalk_env = RandomWalk()
    for i in range(101):
        #print(i)
        print(value.values())
        if i in [0, 1, 10, 100]:
            plt.plot(list(value.values())[1:6], label=str(i))
            plt.plot([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6], label="true values", linewidth=3, color='black')
            plt.xticks([0, 1, 2, 3, 4], ["A", "B", "C", "D", "E"])
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8])
            plt.xlabel("State")
            plt.ylabel("Estimated Value")
            plt.legend(bbox_to_anchor=(1, 1), loc=2)
            plt.savefig('random_walk.png')
        sample_episode(randomwalk_env)


    ##Figure 6.2 right
    fig, ax = plt.subplots(figsize=(10, 8))
    fontsize = 15
    num_episode_array = np.arange(1, 101)
    alpha=0.1
    for alpha in (0.15,0.1,0.05):
        value = {'@': 0, 'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.5, 'E': 0.5, 'F': 0}
        error_values={}
        print(alpha)
        errors = np.zeros(100)

        for run in range(1,101):
            print("""running for {0} time""".format(run))
            value = {'@': 0, 'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.5, 'E': 0.5, 'F': 0}

            for i in range(1,101):
                print("""Episode : {0}""".format(i))

                sample_episode(randomwalk_env,alpha)
                error = rms(list(value.values())[1:6],[1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
                error_values[i] = error_values.get(i,0)+error
        plot_values = [v/100 for v in error_values.values()]
        ax.plot(num_episode_array,
                plot_values,
            linewidth=2,
            linestyle='--',
            label='TD alpha=' + str(alpha))

        ax.set_xlim([min(num_episode_array), max(num_episode_array)])
        ax.grid(linestyle='--')
        ax.legend(loc='best', fontsize=fontsize)
        ax.set_xlabel('Number of episode', fontsize=fontsize)
        ax.set_ylabel('RMS error', fontsize=fontsize)
        ax.set_title('RMS error averaged over states and 100 realizations', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        plt.savefig('random_walk_rms.png')




##getting true values
##https://datascience.stackexchange.com/questions/40899/reinforcement-learning-how-are-these-state-values-in-mrp-calculated
#A:1/6 -0.16666,B:0.333,C=0.5,D=0.66,E=0.83



