##Human vs human
##KID tic tac toe version
##@author BhoooBhoos
import random
import numpy as np
import pickle

class TicTacToe:
    def __init__(self):
        self.board_elements = np.array(['' for i in range(0,10)])
        self.state_action_X = {}
        self.state_action_reward_X = {}

    def draw_board(self):
        print(self.board_elements[7]+' | '+self.board_elements[8]+' | '+self.board_elements[9])
        print('--------')
        print(self.board_elements[4] + ' | ' + self.board_elements[5] + ' | ' + self.board_elements[6])
        print('--------')
        print(self.board_elements[1] + ' | ' + self.board_elements[2] + ' | ' + self.board_elements[3])


    def inputPlayerLetter(self):
        letter =raw_input("Enter the letter you want to play with (O  or X) ")
        opponent_letter = 'X' if letter =='O' else 'O'
        return letter, opponent_letter

    def whoGoesFirst(self):
        return 'X' if random.randint(0,1)==0 else 'O'

    def playAgain(self):
        return raw_input('Do you want to play again')


    def makeMove(self,move,letter,iter):
        if letter=='X':
            if iter in self.state_action_X.keys():
                #print("Value for this iteration till now {0}".format(self.state_action_X[iter] ))
                #print(type(self.state_action_X[iter]))
                self.state_action_X[iter].append((tuple(self.board_elements),move))
                #print("Value after update is {0}".format(self.state_action_X[iter]))
            else:
                #print("Assigning value for first time in this iteration")
                self.state_action_X[iter] = [(tuple(self.board_elements),move)]
                #print("Now the value has been updated for the first time to {0}".format(self.state_action_X[iter]))
        self.board_elements[move]=letter

    def isWinner(self,letter):
        # winner_configurations = [[1,2,3],[4,5,6],[7,8,9],[1,5,9],[3,5,7]]
        # user_configuration = np.where(self.board_elements==letter)[0].tolist()
        # print(user_configuration)
        # return True if user_configuration in winner_configurations else False
        return((self.board_elements[1]==letter and self.board_elements[2]==letter and self.board_elements[3]==letter) or
               (self.board_elements[4] == letter and self.board_elements[5] == letter and self.board_elements[
                   6] == letter) or
               (self.board_elements[7] == letter and self.board_elements[8] == letter and self.board_elements[
                   9] == letter) or
               (self.board_elements[1] == letter and self.board_elements[5] == letter and self.board_elements[
                   9] == letter) or
               (self.board_elements[3] == letter and self.board_elements[5] == letter and self.board_elements[
                   7] == letter) or
               (self.board_elements[1] == letter and self.board_elements[5] == letter and self.board_elements[
                   9] == letter) or
               (self.board_elements[1] == letter and self.board_elements[4] == letter and self.board_elements[
                   7] == letter) or
               (self.board_elements[2] == letter and self.board_elements[5] == letter and self.board_elements[
                   8] == letter) or
               (self.board_elements[3] == letter and self.board_elements[6] == letter and self.board_elements[
                   9] == letter))



    def isSpaceFree(self,move):
        return(True if self.board_elements[move]=='' else False)

    def getPlayerMove(self):
        move =''
        while move not in range(1,10) or not self.isSpaceFree(move):
            move = int(raw_input('Where do you want to place your move [1,9]'))
        return(move)


    def getComputerMove(self):
        move = random.randint(1,10)
        while move not in range(1, 10) or not self.isSpaceFree(move):
            move = random.randint(1,10)
        return (move)


    def getOptimalPolicyMove(self):
        print(self.board_elements)
        print([i for i in range(1,10) if self.board_elements[i] ==''])
        print([(self.state_action_reward_X[(tuple(self.board_elements),i)],i)  for i in range(1,10) if self.board_elements[i] ==''])
        print(sorted([(self.state_action_reward_X[(tuple(self.board_elements),i)],i)  for i in range(1,10) if self.board_elements[i] ==''],reverse=True,key = lambda  tup:tup[0]))
        return sorted([(self.state_action_reward_X[(tuple(self.board_elements),i)],i)  for i in range(1,10) if self.board_elements[i] ==''],reverse=True,key = lambda  tup:tup[0])[0][1]

    def clearBoard(self):
        self.board_elements = np.array(['' for i in range(0,10)])

    def updateReward(self,iter,reward):
        for state_action in self.state_action_X[iter]:
            #print("State action type is {0}".format(type(state_action)))
            #print("State action is {0}".format(state_action))



            if state_action in self.state_action_reward_X.keys():
                self.state_action_reward_X[state_action] = self.state_action_reward_X[state_action] + reward
            else:
                self.state_action_reward_X[state_action] = reward
        # if self.state_action_X[iter] in self.state_action_reward_X.keys():
        #     self.state_action_reward_X[self.state_action_X[iter]] = self.state_action_reward_X[self.state_action_X[iter]]+reward
        # else:
        #     self.state_action_reward_X[self.state_action_X[iter]] = reward
    def upload_pre_read_dict(self):
        pickle_in = open("dict.pickle", "rb")
        self.state_action_reward_X = pickle.load(pickle_in)

if __name__ == '__main__':
    tic_tac_toe = TicTacToe()
    #tic_tac_toe.draw_board()
    #player_letters = tic_tac_toe.inputPlayerLetter()
    player_letters = ['X','O']
    #print(player_letters)
    #print(tic_tac_toe.whoGoesFirst())
    human_counter =0
    computer_counter = 0
    tie_counter = 0
    num_exploration_iterations = 200000
    do_exploration = False
    #print(tic_tac_toe.playAgain())
    ##move = int(raw_input('Where do you want to place your move [1,9]'))
    for i in range(1,num_exploration_iterations+201):
        if(i<num_exploration_iterations):
            if(do_exploration==False):
                continue;
            print(i)
            first_player = tic_tac_toe.whoGoesFirst()
            second_player = 'X' if first_player=='O' else 'O'
          #  print("first_player"+first_player+" second player"+second_player)
            isSpaceLeft = 1
            while(not tic_tac_toe.isWinner(first_player) and not tic_tac_toe.isWinner(second_player) and isSpaceLeft==1):
                #move = tic_tac_toe.getPlayerMove()
                #print("Human move")
                move = tic_tac_toe.getComputerMove()


                tic_tac_toe.makeMove(move,first_player,i)
                #tic_tac_toe.draw_board()

                if(tic_tac_toe.isWinner(player_letters[0]) or not (True in [tic_tac_toe.isSpaceFree(move) for move in range(1,10)])):
                    break;
                #print("Your move "+str(move))

                #print("Now my move human")
                computer_move = tic_tac_toe.getComputerMove()
                tic_tac_toe.makeMove(computer_move, second_player,i)
                #tic_tac_toe.draw_board()
                isSpaceLeft = 1 if True in [tic_tac_toe.isSpaceFree(move) for move in range(1,10)] else 0
                #print('Tie' if isSpaceLeft==0 else 'Continue')

            if( tic_tac_toe.isWinner(player_letters[0])==True):
                human_counter = human_counter+1
                tic_tac_toe.updateReward(i, 10)
                #print('You won human')
                ##update state counts for winner by +1 to all states,action in X
            elif( tic_tac_toe.isWinner(player_letters[1])==True) :
                computer_counter = computer_counter+1
                tic_tac_toe.updateReward(i, -10)
                ##update state counts for loser X by -1 to all states,action in X
                #print('Computer beats human')
            else:
                tie_counter = tie_counter+1
                #print("Tie")
                tic_tac_toe.updateReward(i, 0)
            #print(tic_tac_toe.state_action_X[i])
            tic_tac_toe.clearBoard()
        elif i==   num_exploration_iterations:
            if(do_exploration==False):
                continue;

            print("saving policy to file")
            pickle_out = open("dict.pickle", "wb")
            pickle.dump(tic_tac_toe.state_action_reward_X, pickle_out)
            pickle_out.close()

        else:
            print(i)
            tic_tac_toe.upload_pre_read_dict()
            first_player = tic_tac_toe.whoGoesFirst()
            second_player = 'X' if first_player == 'O' else 'O'
            #  print("first_player"+first_player+" second player"+second_player)
            isSpaceLeft = 1
            while (not tic_tac_toe.isWinner(first_player) and not tic_tac_toe.isWinner(
                    second_player) and isSpaceLeft == 1):
                # move = tic_tac_toe.getPlayerMove()
                # print("Human move")
                move = tic_tac_toe.getOptimalPolicyMove() if first_player=='X' else tic_tac_toe.getComputerMove()

                tic_tac_toe.makeMove(move, first_player, i)
                tic_tac_toe.draw_board()

                if (tic_tac_toe.isWinner(player_letters[0]) or not (
                    True in [tic_tac_toe.isSpaceFree(move) for move in range(1, 10)])):
                    break;
                # print("Your move "+str(move))

                # print("Now my move human")
                computer_move =  tic_tac_toe.getOptimalPolicyMove() if second_player=='X' else tic_tac_toe.getComputerMove()
                tic_tac_toe.makeMove(computer_move, second_player, i)
                tic_tac_toe.draw_board()
                isSpaceLeft = 1 if True in [tic_tac_toe.isSpaceFree(move) for move in range(1, 10)] else 0
                # print('Tie' if isSpaceLeft==0 else 'Continue')

            if (tic_tac_toe.isWinner(player_letters[0]) == True):
                human_counter = human_counter + 1
                tic_tac_toe.updateReward(i, 15)
                # print('You won human')
                ##update state counts for winner by +1 to all states,action in X
            elif (tic_tac_toe.isWinner(player_letters[1]) == True):
                computer_counter = computer_counter + 1
                tic_tac_toe.updateReward(i, -10)
                ##update state counts for loser X by -1 to all states,action in X
                # print('Computer beats human')
            else:
                tie_counter = tie_counter + 1
                # print("Tie")
                tic_tac_toe.updateReward(i, 0)
            # print(tic_tac_toe.state_action_X[i])


            tic_tac_toe.clearBoard()

    print(human_counter)
    print(computer_counter)
    print(tie_counter)
    # print(tic_tac_toe.state_action_reward_X)
        # move = int(raw_input('Where do you want to place your move [1,9]'))
    # tic_tac_toe.makeMove(move,player_letters[0])
    # move = int(raw_input('Where do you want to place your move [1,9]'))
    # tic_tac_toe.makeMove(move,player_letters[0])
#    tic_tac_toe.isWinner(player_letters[0])


