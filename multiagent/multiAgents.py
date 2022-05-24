# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
import heapq
from util import manhattanDistance
from game import Directions
import random, util
from searchAgent import calculate_ghost_distance, calculate_maze_distance
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState, previous_state, is_stop):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        print("self.depth: ", self.depth)
        print("self.evaluationFunction: ", self.evaluationFunction)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def alphabeta(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                if action == 'Stop':
                    is_stop = True
                else:
                    is_stop = False
                succ  = minValue(state, state.generateSuccessor(0, action), 1, 1, is_stop)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(previous_state, state, agentIdx, depth, is_stop):
            if agentIdx == state.getNumAgents():
                return maxValue(previous_state, state, 0, depth + 1, is_stop)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(previous_state, state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, is_stop)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state, previous_state, is_stop)


        def maxValue(previous_state, state, agentIdx, depth, is_stop):
            if depth > self.depth:
                return self.evaluationFunction(state, previous_state, is_stop)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(previous_state, state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, is_stop)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state, previous_state, is_stop)

        action = alphabeta(gameState)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def alphabeta(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            alpha = float("-Inf")
            beta = float("Inf")

            
            for action in state.getLegalActions(0):
                if action == 'Stop':
                    is_stop = True
                else:
                    is_stop = False
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state, state.generateSuccessor(0, action), 1, 1, alpha, beta, is_stop)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action

                # if bestValue >= beta:                         # never happened
                #     break
                
                alpha = max(bestValue, alpha)

            print(value)
            return bestAction

        def minValue(previous_state, state, agentIdx, depth, alpha, beta, is_stop):
            if agentIdx == state.getNumAgents():
                return maxValue(previous_state, state, 0, depth + 1, alpha, beta, is_stop)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(previous_state, state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta, is_stop)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

                if value <= alpha:
                    return value
                
                beta = min(value, beta)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state, previous_state, is_stop)


        def maxValue(previous_state, state, agentIdx, depth, alpha, beta, is_stop):
            if depth > self.depth:
                return self.evaluationFunction(state, previous_state, is_stop)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(previous_state, state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta, is_stop)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)

                if value >= beta:
                    return value
                
                alpha = max(value, alpha)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state, previous_state, is_stop)

        action = alphabeta(gameState)
        print(action)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def alphabeta(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                if action == 'Stop':
                    is_stop = True
                else:
                    is_stop = False

                succ  = minValue(state, state.generateSuccessor(0, action), 1, 1, is_stop)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(previous_state, state, agentIdx, depth, is_stop):
            if agentIdx == state.getNumAgents():
                return maxValue(previous_state, state, 0, depth + 1, is_stop)
            value = None

            num_of_actions = len(state.getLegalActions(agentIdx))
            
            for action in state.getLegalActions(agentIdx):
                action_prob =  1/num_of_actions
                succ = minValue(previous_state, state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, is_stop) * action_prob
                if value is None:
                    value = succ
                else:
                    value += succ

            if value is not None:
                return value 
            else:
                return self.evaluationFunction(state, previous_state, is_stop)


        def maxValue(previous_state, state, agentIdx, depth, is_stop):
            if depth > self.depth:
                return self.evaluationFunction(state, previous_state, is_stop)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(previous_state, state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, is_stop)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state, previous_state, is_stop)

        action = alphabeta(gameState)
        print(action)
        return action

import numpy as np 


def betterEvaluationFunction(currentGameState, previous_state, is_stop):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    oldPos = previous_state.getPacmanPosition()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # ghost_position = [currentGameState.getGhostPosition(i) for i in range(1, currentGameState.getNumAgents())]
    pacman_ghost_distance = [calculate_ghost_distance(currentGameState, i) for i in range(1, currentGameState.getNumAgents())]  
    game_score = currentGameState.getScore()

    total_scare_time = sum(newScaredTimes)
    capsule_count = len(newCapsules)

    if capsule_count > 0 and total_scare_time == 0:
        avg_capsule = sum([calculate_maze_distance(currentGameState, caps) for caps in newCapsules])/capsule_count
    else:
        avg_capsule = 0.00001                                                                     # if you eat all the capsule, you will have a small reward



    foodList = newFood.asList()
    food_count = len(foodList)
    
    if food_count > 0:
        closest_food = min([calculate_maze_distance(currentGameState, food) for food in foodList])
    else:                                                                                  
        closest_food = 0.00001                                                             
        avg_capsule /= 10

    
    hunter_ratio = np.zeros(currentGameState.getNumAgents() - 1)
    for i, scare_time in enumerate(newScaredTimes):
        distance = pacman_ghost_distance[i]
        if scare_time == 0:
            if distance < 1:
                hunter_ratio[i] = -10           # run        


    hunter_ratio = np.sum(hunter_ratio)


    if is_stop:
        stop_penalty = -10
    else:
        stop_penalty = 0


    features = [game_score,
                1/closest_food, 
                1/avg_capsule,
                capsule_count,
                hunter_ratio,
                stop_penalty
                ]

    weights = [5,
               30,
               10,
               -10,
               40, 
               20]


    return  sum([feature * weight for feature, weight in zip(features, weights)])


# Abbreviation
better = betterEvaluationFunction



