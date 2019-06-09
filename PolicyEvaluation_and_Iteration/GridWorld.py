##Policy Evaluation
##Evaluate how good a policy is, how much total reward we will get if we take action a given by the policy in state s
##value of each state under a given policy
##if we follow the policy how good it is to be in this state
##Computes value function of a policy  using Bellman expectation equation


##Problem : Evaluate a given policy
##Solution: iterative application of Bellman Expectation equation

import numpy as np

class GridWorldEnvironment:
    ##class to model gridworld environment

    def __init__(self,cells_per_row =4):
        self.start_column = 0
        self.end_column = 3
        self.gridworldsize = cells_per_row ** 2
        self.states = range(0,self.gridworldsize)
        self._cells_per_row = cells_per_row
        self.actions = ['left','right','up','down']
        self.discount =1

    def is_terminal_state(self,state):
        return state == 0 or state == self.gridworldsize-1

    def get_next_state(self,current_state,action):
        ##get the next state given current state and action taken
        ##for grid world it is deterministic
        if action == 'left':
            next_state = current_state-1 if current_state%self._cells_per_row != 0 else current_state
        elif action == 'right':
            next_state = current_state + 1 if (current_state+1)%self._cells_per_row !=0 else current_state
        elif action == 'up':
            next_state = current_state-self._cells_per_row if current_state-self._cells_per_row>=0 else current_state
        elif action == 'down':
            next_state = current_state + self._cells_per_row if current_state + self._cells_per_row < self.gridworldsize else current_state
        elif self.is_terminal_state(current_state):
            next_state = current_state
        return next_state
    def reward(self,current_state):
        if self.is_terminal_state(current_state)==True:
            return 0
        else:
            return -1


class RandomPolicy:
    ##class to model random policy
    def __init__(self,environment):
        self.env = environment

    def action(self,current_state):

        ##return probabilities associated with each action given a state
        if self.env.is_terminal_state(current_state) == True:
            return {'no_action':1}
        else:
            return {action:1/len(self.env.actions) for action in self.env.actions}


class GreedyPolicy:

    def __init__(self,environment,policy_values):
        self.env = environment
        self.policy_values = policy_values

    def action(self,current_state):

        ##return probabilities associated with each action given a state
        if self.env.is_terminal_state(current_state):
            return {'no_action':1}
        else:
            state_possible_values = [
                (possible_action, self.policy_values[environment.get_next_state(current_state, possible_action)]) for possible_action
                in
                self.env.actions]
            optimal_action = sorted(state_possible_values, key=lambda tup: tup[1], reverse=True)[0][0]
            return {action: 1 if action == optimal_action else 0 for action in ['left', 'right', 'up', 'down']}



def policy_evaluation(numIterations,environment,policy):
    value = np.zeros(environment.gridworldsize)
    print("Initial values :{0}".format(value))
    for iteration in range(numIterations):
        new_value = np.zeros(environment.gridworldsize)
        print("Iteration : {0}".format(iteration))
        for state in environment.states:
            for possible_action, possible_prob in policy.action(state).items():
                new_value[state] = new_value[state] + possible_prob * (environment.reward(state) + (
                environment.discount * value[environment.get_next_state(state, possible_action)]))
        value = new_value

        print(np.trunc(value * 10) / 10)
    return(np.round(value, 0))




if __name__ == '__main__':
    numIterations = 10
    environment = GridWorldEnvironment()
    policy = RandomPolicy(environment)

    for i in range(2):
        policy_value = policy_evaluation(numIterations, environment, policy)


        print("Value of policy after policy evaluation loop")
        print(policy_value)
        policy = GreedyPolicy(environment,policy_value)

    for state in environment.states:
        print(policy.action(state))

