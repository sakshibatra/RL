##Model Free control : Policy evaluation and policy iteration
##Policy evaluation : estimating q(s,a) using monte carlo simulations
##Policy iteration  : choosing the action a for each state s which maximizes q(s,a)

##To enable exploration; we use exploration starts  in which we start from a random state action pair
import gym
import numpy as np
from functools import reduce

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import seaborn as sns

##for each of the face card value is 10
deck = [1,2,3,4,5,6,7,8,9,10,10,10,10]

def takeSecond(elem):
    return elem[1]

def draw_card():
    """
    draw a single card from the deck
    :return:
    """
    return np.random.choice(deck)

def draw_hand():
    """
    draw cards for a given seed
    :return: two cards at random
    """
    return[np.random.choice(deck),np.random.choice(deck)]

def usable_ace(hand):
    """

    :return: if the hand has ace in it
    """
    return 1 in hand and sum(hand)+10<=21
    #return 1 in hand

def sum_hand(hand):
    """
    sum of cards in hand
    if there is an Ace such sum of the cards is less than 21 then it is taken as 11
    else it is counted as 1
    :param hand:
    :return:
    """
    if(usable_ace(hand)):
        return sum(hand)+10
    else:
        return sum(hand)

def is_game_over(hand):
    """

    :return: if sum
    """
    return 1 if np.sum(hand) >21 else 0

class BlackJack:

    def __init__(self):
        ##Two kinds of action :
            # stick - stop getting cards
            # twist - ask for another card
            ##starting at random state and action
        self.action_space = ['stick', 'twist']
        sum_of_cards = np.random.randint(low=1, high=21, size=1)[0]
        dealer_showing_card = np.random.randint(low=1, high=10, size=1)[0]
        is_usable_ace = np.random.randint(low=0, high=1, size=1)[0]
        self.observation_space = (sum_of_cards, dealer_showing_card, is_usable_ace)
        self.reset()

        # self.action_space = ['stick','twist']
        # sum_of_cards = np.random.randint(low=1,high=21,size = 1)[0]
        # dealer_showing_card = np.random.randint(low=1,high=10,size = 1)[0]
        # is_usable_ace = np.random.randint(low=0,high=1,size = 1)[0]
        # ace = True if ((is_usable_ace==1) and sum_of_cards+10<=21) else False
        # self.observation_space = (sum_of_cards,dealer_showing_card,ace)

    def update_state(self):
        """
        update the observation space for player
        :return:  observation space
        """
        return(sum_hand(self.player),self.dealer[0],usable_ace(self.player))

    def reset(self):
        """
        Start with a random state to enable exploration
        """
        # sum_of_cards = np.random.randint(low=1, high=21, size=1)[0]
        # dealer_showing_card = np.random.randint(low=1, high=10, size=1)[0]
        # is_usable_ace = np.random.randint(low=0, high=1, size=1)[0]
        # ace = True if ((is_usable_ace == 1) and sum_of_cards + 10 <= 21) else False
        #
        # return (sum_of_cards,dealer_showing_card,ace)

        self.player = draw_hand()
        self.dealer = draw_hand()
        return self.update_state()




    def play_move(self,action):
        """

        :param action: action that player chooses to make; one of twist/stick
        :return: reward associated with taking that action

        """
        assert action in self.action_space
        if action =='stick':
            ##Stop getting card and play out dealer card
            done = True;
            ##while the dealer card sum is less than 17 , he will keep on getting cards
            while sum_hand(self.dealer)<17:
                self.dealer.append(draw_card())

            if  sum_hand(self.player)       >    sum_hand(self.dealer) or sum_hand(self.dealer)>21 :
                reward = 1
            elif sum_hand(self.player)       ==   sum_hand(self.dealer):
                reward = 0
            else :
                reward = -1

        else:
            ##if action is twist
            ##take another card
            self.player.append(draw_card())
            if is_game_over(self.player):
                reward =-1
                done = True
            else:
                done = False
                reward = 0


        return self.update_state(),done,reward


##define policy

class NaiveBJ:
    def __init__(self,environment):
        self.env = environment

    def action(self,state):
        """

        :return: action following naive policy
        stick if sum of cards ≥ 20, otherwise twist
        """
        if state[0]>=20:
            return 'stick'
        else:
            return 'twist'

class GreedyBJ:
    def __init__(self,environment,policy_values):
        self.env = environment
        self.policy_values = policy_values
        self.epsilon = 0.1
    def action(self,state):
        """

        :return: action following naive policy
        with prob 1- epsilon using the policy values for the state choose the action which maximizes q(s,a)
        with prob epsilon
        """

        action_values = [(i[0][1], i[1]) for i in result.items() if i[0][0] == state]
        if(len(action_values)==0):
            chosen_action = random.sample(['stick', 'twist'], 1)[0]
        elif((len(action_values)==1) and (action_values[0][1]<0)):
            chosen_action = 'stick' if action_values[0][0]=='twist' else 'twist'
        else:
            best_action =sorted(action_values,key= takeSecond,reverse=True)[0][0]
            chosen_action = best_action if np.random.uniform(0, 1, 1) > self.epsilon else random.sample(['stick','twist'],1)[0]

        return chosen_action

def play_episode(environment,policy):
    """
    play a sample episode of the game
    :return: (state,action, reward)
    """
    episode = []
    state = environment.reset();
    # action_chosen = random.sample(['stick','twist'],1)[0]
    # new_state, done, reward = environment.play_move(action_chosen)
    # episode.append((state, action_chosen, reward))
    # state = new_state
    # if done:
    #     return episode

    while True:
        ##take an action based on policy
        action_chosen = policy.action(state)
        new_state,done,reward =  environment.play_move(action_chosen)
        episode.append((state,action_chosen,reward))
        state = new_state
        if done:
            break

    #print(episode)
    return episode




def update_value(episode,discount =1):
    """


    :param episode:
    :param discount:
    :return:
    """
    return_g = 0
    returns ={}
    num_entries={}
    episode.reverse()
    for time_step,(state,action,reward) in enumerate(episode):
        #print(time_step)
        #print(reward)
        return_g = return_g*discount + reward
        #print(return_g)
        if (state,action) not in [(x[0],x[1]) for x in episode[:(len(episode)-time_step -1)]]:
            returns[(state,action)] = return_g

    return(returns)




if __name__ == '__main__':
    environment = BlackJack()
    policy = NaiveBJ(environment)




    value_returns = {}
    num_occ = {}
    result = {}

    for i in range(500000):
        ##in each episode we start with a random state action pair
        print(i)
        episode = play_episode(environment,policy)
        episode_return = update_value(episode)
        #print("""Episode return {0}""".format(episode_return))
        for state_action,reward in episode_return.items():
            value_returns[state_action] = value_returns.get(state_action,0)+reward
            num_occ[state_action] = num_occ.get(state_action,0)+1

        for state_action in value_returns.keys():
            result[state_action] = value_returns.get(state_action)/num_occ.get(state_action)


        policy = GreedyBJ(environment,result)
        ##update policy


    print(result)
    #print("""wtf : {0} """.format(result))

    X=list();Y=list();Z=list()
    X1=list();Y1=list();Z1=list()
    for key,value in result.items():
        if not key[0][2]:
            if key[0][0]>=12:
                X.append(key[0][1])
                Y.append(key[0][0])
                Z.append(value)
        else:
            if key[0][0] >= 12:
                X1.append(key[0][1])
                Y1.append(key[0][0])
                Z1.append(value)
    #
    # print(X)
    # print(Y)
    # print(Z)
    ##plot result

## for each state which is the optimal action

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(X,Y,Z, cmap='RdYlBu', linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('optimal_policy_bj.png')

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(X1, Y1, Z1, cmap='RdYlBu', linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('optimal_policy_bj_2.png')


    policy = GreedyBJ(environment, result)

    X=list();Y=list();Z=list()
    X1=list();Y1=list();Z1=list()
    for key,value in result.items():
        if not key[0][2]:
            if key[0][0]>=12:
                X.append(key[0][1])
                Y.append(key[0][0])
                action_values = [(i[0][1], i[1]) for i in result.items() if i[0][0] == key[0]]

                best_action = sorted(action_values, key=takeSecond,reverse=True)[0][0]

                Z.append(100 if best_action=='stick' else 0)
                print(key[0],policy.action(key[0]))
        else:
            if key[0][0] >= 12:
                X1.append(key[0][1])
                Y1.append(key[0][0])
                action_values = [(i[0][1], i[1]) for i in result.items() if i[0][0] == key[0]]

                best_action = sorted(action_values, key=takeSecond)[0][0]

                Z1.append(100 if best_action=='stick' else 0)



    plot_dim = sorted(set([(x,y,z) for x,y,z in zip(X,Y,Z)]),key=lambda tup:(tup[0],tup[1]))
    X = [x[0] for x in plot_dim]
    Y = [x[1] for x in plot_dim]
    Z = [x[2] for x in plot_dim]

    print(plot_dim)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.scatter3D(X, Y, Z, cmap='RdYlBu', linewidth=0.1)
    plt.savefig('optimal_action_bj.png')

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.scatter3D(X1, Y1, Z1, cmap='RdYlBu', linewidth=0.1)
    plt.savefig('optimal_action_bj_2.png')






                # print(value_returns)
    # occ = {}
    # def dict_aggregator(x, y):
    #     for key, val in x:
    #         y[key] = y.get(key, 0) + val
    #     return y
    #
    # all_keys = set([key.keys() for key in value_returns])
    # print(all_keys)
    # #product = [x for value_return in value_returns for x in value_return]
    # #print(product)
    #
