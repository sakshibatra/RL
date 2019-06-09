import gym
import numpy as np
from functools import reduce

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

deck = [1,2,3,4,5,6,7,8,9,10,10,10,10]

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
    return 1 if np.sum(hand) >=21 else 0

class BlackJack:

    def __init__(self):
        ##Two kinds of action :
            # stick - stop getting cards
            # twist - ask for another card

        self.action_space = ['stick','twist']
        sum_of_cards = np.random.randint(low=1,high=21,size = 1)[0]
        dealer_showing_card = np.random.randint(low=1,high=10,size = 1)[0]
        is_usable_ace = np.random.randint(low=0,high=1,size = 1)[0]
        self.observation_space = (sum_of_cards,dealer_showing_card,is_usable_ace)
        self.reset()

    def update_state(self):
        """
        update the observation space for player
        :return:  observation space
        """
        return(sum_hand(self.player),self.dealer[0],usable_ace(self.player))

    def reset(self):
        """
        draw two cards at random for dealer and player both
        :return: updated space
        """
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

            if  sum_hand(self.player)       >    sum_hand(self.dealer):
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



def play_episode(environment,policy):
    """
    play a sample episode of the game
    :return: (state,action, reward)
    """
    episode = []
    state = environment.reset();
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
        if state not in [x[0] for x in episode[:(len(episode)-time_step -1)]]:
            returns[state] = return_g

    return(returns)




if __name__ == '__main__':
    environment = BlackJack()
    policy = NaiveBJ(environment)
    value_returns = {}
    num_occ = {}
    for i in range(500000):
        episode = play_episode(environment,policy)
        episode_return = update_value(episode)
        #print("""Episode return {0}""".format(episode_return))
        for state,reward in episode_return.items():
            value_returns[state] = value_returns.get(state,0)+reward
            num_occ[state] = num_occ.get(state,0)+1

    # print(value_returns)
    # print(num_occ)
    result ={}
    for states in value_returns.keys():
        result[states] = value_returns.get(states)/num_occ.get(states)

    #print("""wtf : {0} """.format(result))

    X=list();Y=list();Z=list()
    X1=list();Y1=list();Z1=list()
    for key,value in result.items():
        if not key[2]:
            if key[0]>=12:
                X.append(key[1])
                Y.append(key[0])
                Z.append(value)
        else:
            if key[0] >= 12:
                X1.append(key[1])
                Y1.append(key[0])
                Z1.append(value)
    #
    # print(X)
    # print(Y)
    # print(Z)
    ##plot result
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(X,Y,Z, cmap='RdYlBu', linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('blackjack_fig1.png')

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(X1, Y1, Z1, cmap='RdYlBu', linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('blackjack_fig2.png')






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
