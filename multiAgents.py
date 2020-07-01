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


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np
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
        new_food_xys = newFood.asList()
        current_food = currentGameState.getFood().asList()
        tot = 0
        ghosts = []
        food_dists = []

        for xy in current_food:
            food_dist = manhattanDistance(xy, newPos)
            food_dists.append(food_dist)

        for ghost in newGhostStates:
            dist = manhattanDistance(newPos, ghost.getPosition())
            scared_time = ghost.scaredTimer
            ghosts.append((dist, scared_time))

        closest_ghost = min(ghosts, key=lambda g: g[0])
        closest_food = min(food_dists)

        # Case: ghost is not scared and really close -> run away!
        if closest_ghost[0] < 2 and closest_ghost[1] < 1:
            tot -= 10

        # Don't want to stop unless pacman is going to die.
        if action == 'Stop':
            tot -= 4

        # if there is food to be eaten, eat it!
        if len(new_food_xys) < len(current_food):
            tot += 9
        elif closest_food < 2:
            tot += 8
        elif closest_food < 3:
            tot += 7
        elif closest_food < 4:
            tot += 6
        elif closest_food < 5:
            tot += 5
        elif closest_food < 10:
            tot += 4
        elif closest_food < 15:
            tot += 3
        elif closest_food < 20:
            tot += 2
        else:
            tot += 1

        return tot






def scoreEvaluationFunction(currentGameState):
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


    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)



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
        # just like the lecture slides only pair values and actions together with twoples.
        num_ghosts = gameState.getNumAgents() - 1

        def max_value(state, depth, agentIndex):

            # check terminal states
            if depth < 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None

            v = (float("-inf"), None)
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, (min_value(successor, depth, agentIndex+1)[0], action), key= lambda x: x[0])

            return v


        def min_value(state, depth, agentIndex):

            # check terminal states
            if depth < 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None

            v = (float("inf"), None)

            # Case: agent is a ghost
            if agentIndex < num_ghosts:
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex,action)
                    v = min(v, (min_value(successor, depth, agentIndex+1)[0], action), key= lambda x: x[0])

            # Case: agent is pacman
            else:
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex,action)
                    v = min(v, (max_value(successor, depth-1, 0)[0], action), key= lambda x: x[0])

            return v

        # start on rood which is pacman
        return max_value(gameState, self.depth-1, 0)[1]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_ghosts = gameState.getNumAgents() - 1

        def max_value(state, depth, agentIndex, alpha, beta):

            # Check terminal states
            if depth < 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None

            v = (float("-inf"), None)
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, (min_value(successor, depth, agentIndex+1, alpha, beta)[0], action), key= lambda x: x[0])

                # prune
                if v[0] > beta:
                    return v
                alpha = max(alpha, v[0])

            return v


        def min_value(state, depth, agentIndex, alpha, beta):

            # check terminal states
            if depth < 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None

            v = (float("inf"), None)

            # Case: agent is a ghost
            if agentIndex < num_ghosts:
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex,action)
                    successor_val = (min_value(successor, depth, agentIndex+1, alpha, beta)[0], action)

                    # prune
                    if successor_val[0] < v[0]:
                        v = successor_val
                    if v[0] < alpha:
                        return v
                    beta = min(beta, v[0])

            # Case: agent is pacman
            else:
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex,action)
                    successor_val =(max_value(successor, depth-1, 0, alpha, beta)[0], action)

                    # prune
                    v = min(v, successor_val, key= lambda x: x[0])
                    if successor_val[0] < v[0]:
                        v = successor_val
                    if v[0] < alpha:
                        return v
                    beta = min(beta, v[0])

            return v

        return max_value(gameState, self.depth-1, 0, float("-inf"), float("inf"))[1]



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(state, depth, agentIndex):
            if depth == 0 and agentIndex == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None
            if agentIndex == 0:
                return max_value(state, depth - 1, agentIndex)
            else:
                return exp_value(state, depth, agentIndex)

        # maximize pacman
        def max_value(state, depth, agentIndex):
            v = (float("-inf"), None)
            legal_actions = state.getLegalActions(agentIndex)

            if len(legal_actions) == 0:
                return value(state, 0, 0)

            for action in legal_actions:
                successor_val = (value(state.generateSuccessor(agentIndex, action), depth, agentIndex+1)[0], action)
                v = max(v, successor_val, key=lambda x: x[0])

            return v

        # minimize ghosts
        # return a tuple to be consistent with value()
        def exp_value(state, depth, agentIndex):
            legal_actions = state.getLegalActions(agentIndex)
            values = []

            if len(legal_actions) == 0:
                return value(state, 0, 0)

            for action in legal_actions:
                v = value(state.generateSuccessor(agentIndex, action), depth, (agentIndex+1) % state.getNumAgents())[0]
                values.append(v)

            return np.mean(values), None

        return value(gameState, self.depth, 0)[1]



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    ghost_states = currentGameState.getGhostStates()
    pacman_xys = currentGameState.getPacmanPosition()
    food_xys = currentGameState.getFood().asList()
    power_pellet_xys = currentGameState.getCapsules()
    ghosts = []
    food_pts = 500 / (1 + len(food_xys))
    cap_pts = 1000 / (1 + len(power_pellet_xys))

    # find closest ghost so it can be avoided.
    for ghost in ghost_states:
        dist = manhattanDistance(pacman_xys, ghost.getPosition())
        scared_time = ghost.scaredTimer
        ghosts.append((dist, scared_time))

    closest_ghost = min(ghosts, key=lambda g: g[0])
    ghost_pts = 1 / (1 + float(closest_ghost[0]))

    # case: ghost is close and scared -> get it if it's easy.
    if closest_ghost[0] < 1 and closest_ghost[1] > 0:
        ghost_pts += 10

    # case: ghost is close and not scared -> run away!
    if closest_ghost[0] < 1 and closest_ghost[1] < 1:
        ghost_pts += float('-inf')

    return food_pts + cap_pts + ghost_pts



# Abbreviation
better = betterEvaluationFunction


def manhattanDistance(xy1, xy2):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])




