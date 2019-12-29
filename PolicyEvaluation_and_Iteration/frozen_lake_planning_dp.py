##Openai gym

import gym
import pandas as pd
from gym.envs.toy_text import frozen_lake
import itertools
import numpy as np

env = gym.make("FrozenLake-v0")
env.reset()
env.render()
#action space
print(env.action_space) #0 to 3

print(env.observation_space) #0 to 15


##Creating a bigger version of openai frozen lake
a=np.array(list(itertools.repeat(frozen_lake.MAPS['8x8'],4)))
a[1:,0]=np.char.replace(a[1:,0], 'S','H')
a[:-1,-1]=np.char.replace(a[:-1,-1], 'G','F')
a=np.concatenate(a)
a=a.reshape(16,2)
a=[i+j for i,j in a]

frozen_lake.MAPS['16x16']=a
import gym
gym.envs.register(
    id='FrozenLake16x16-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '16x16'},
    max_episode_steps=600,
    reward_threshold=0.99, # optimum = 1
)
env = gym.make("FrozenLake16x16-v0", is_slippery=False)
#env = gym.make("FrozenLake-v0")
env.reset()
env.render()
# num_tries = 30
# max_iterations = 100
#
# for i in range(num_tries):
#     for i in range(max_iterations):
#         chosen_action = env.action_space.sample()
#         observation,reward,done,info = env.step(chosen_action)
#         env.render()
#         if done:
#             break

##Not so good !!


##Setting deterministic mode for frozen lake
#env = gym.make("FrozenLake8x8-v0", is_slippery=False)

#
# env.reset()
# env.render()
# num_tries = 30
# max_iterations = 100
#
# for i in range(num_tries):
#     for i in range(max_iterations):
#         chosen_action = env.action_space.sample()
#         observation,reward,done,info = env.step(chosen_action)
#         env.render()
#         if done:
#             break


##Goal : to reach Goal state(G) from start state(S) in minimum number of turns

##policy evaluation
import numpy as np
V = np.zeros(env.observation_space.n)

def evaluate_policy(env,V,pi,gamma,theta):
    """

    :param env:
    :param V:
    :param pi:
    :param gamma:
    :param theta:
    :return:
    """
    while True:
        delta = 0

        for state in range(env.observation_space.n):
            v = V[state]
            bellman_update_value_fn(env, V, pi[state], state, gamma)
            delta = max(delta, abs(v - V[state])) ##difference between value fn state in current iteration and last iteration
        if delta<theta:
            break

    return V


def bellman_update_value_fn(env, V, pi, s, gamma):
    """

    :param env:
    :param V:
    :param pi:prob disribution oer action space
    :param s:
    :param gamma:
    :return:
    """

    v = 0
    for action,prob_action in enumerate(pi):
        for prob_next_state, next_state, reward_next_state, done in env.P[s][action]:
            v+=prob_action*prob_next_state*(reward_next_state+gamma*V[next_state])


    V[s] = v




##random policy : uniform distribution ever action would have same prob
policy = np.ones((env.nS, env.action_space.n) )/ env.action_space.n
theta =1e-8
gamma = 0.9
evaluate_policy(env,V,policy,gamma,theta)
env.render()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(32, 32))
sns.heatmap(V.reshape(16, 16),  cmap="YlGnBu", annot=True, cbar=False);
plt.show()



##Policy improvement

def improve_policy(env, V, pi, gamma):
    policy_stable = True
    for s in range(env.observation_space.n):
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        if not np.array_equal(pi[s], old):
            policy_stable = False
    return pi, policy_stable

def policy_iteration(env, gamma, theta):
    V = np.zeros(env.nS)
    pi = np.ones((env.nS, env.action_space.n) )/ env.action_space.n
    policy_stable = False
    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
    return V, pi


def q_greedify_policy(env, V, pi, s, gamma):
    """Mutate ``pi`` to be greedy with respect to the q-values induced by ``V``."""
    ### START CODE HERE ###
    ##q(s,a)=sigma(P(ss')*(gamma*V(s')+R(s,a,s'))
    q = np.zeros((env.action_space.n))
    for idx, action in enumerate(range(env.action_space.n)):
        for prob_next_state, next_state, reward_next_state, done in env.P[s][action]:
            q[idx] += prob_next_state * ((gamma * V[next_state]) + reward_next_state)

    greedy_action = np.argmax(q)
    # print(greedy_action)
    for action, action_prob in enumerate(pi[s]):
        if action == greedy_action:
            print(action, greedy_action)
            pi[s][action] = 1
        else:
            pi[s][action] = 0


V, pi = policy_iteration(env, gamma, theta)

env.reset()
env.render()
initial_state = 0
steps = 0
step_matrix = np.zeros(len(env.P))
while True:

    env.step(np.argmax(pi[initial_state]))
    env.render()
    step_matrix[initial_state] = steps
    steps = steps + 1
    prob_next_state, initial_state, reward_next_state, done = env.P[initial_state][np.argmax(pi[initial_state])][0]
    if steps%10==0:
        print(steps)
    if initial_state ==255:
        print("done in {0} steps".format(steps))
        break


plt.figure(figsize=(32, 32))
sns.heatmap(V.reshape(16, 16),  cmap="YlGnBu", annot=True, cbar=False);
plt.show()

##plotting path used
annotations = np.array([list(i) for i in frozen_lake.MAPS['16x16']]).reshape(1,256)
annotations_ = annotations[0].tolist()
p=pd.DataFrame(np.array(['X' if j=='H' else str(i) for i,j in zip(step_matrix.astype(int).tolist(),annotations_)]).reshape(16,16))
