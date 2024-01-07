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
from cmath import inf

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        newFood = currentGameState.getFood()

        # czy trzeba uciekać
        for ghostPos in successorGameState.getGhostPositions():
            if util.manhattanDistance(ghostPos, newPos) <= 1:
                return -inf

        # czy jedzenie obok
        if newFood[newPos[0]][newPos[1]]:
            return float('inf')

        # dystans do jedzenia
        minDist = -inf

        for food in newFood.asList():
            dist = util.manhattanDistance(food, newPos) * -1
            if dist > minDist:
                minDist = dist

        return minDist


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        def get_max(gameState, curr_depth):
            if (gameState.isWin() or gameState.isLose() or curr_depth == 0):  # base case
                return self.evaluationFunction(gameState)

            max_value = -inf
            for action in gameState.getLegalActions(0):
                current = get_min(gameState.generateSuccessor(0, action), curr_depth - 1, 1)
                if (current > max_value):
                    max_value = current
            return max_value

        def get_min(gameState, curr_depth, agent):
            if (gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)

            min_value = inf
            for action in gameState.getLegalActions(agent):

                if (agent == gameState.getNumAgents() - 1):
                    current = get_max(gameState.generateSuccessor(agent, action), curr_depth)
                else:
                    current = get_min(gameState.generateSuccessor(agent, action), curr_depth, agent + 1)

                if (current < min_value):
                    min_value = current

            return min_value

        max_value = -inf
        best_action = None

        for action in gameState.getLegalActions(0):

            current = get_min(gameState.generateSuccessor(0, action), self.depth - 1, 1)

            if current > max_value:
                max_value = current
                best_action = action

            if best_action == None:
                best_action = action

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def get_max(gameState, curr_depth, alpha, beta):
            if (gameState.isWin() or gameState.isLose() or curr_depth == 0):
                return self.evaluationFunction(gameState)

            max_value = -inf
            for action in gameState.getLegalActions(0):
                current = get_min(gameState.generateSuccessor(0, action), curr_depth - 1, 1, alpha, beta)

                if max_value < current:
                    max_value = current

                if max_value > alpha:
                    alpha = max_value

                if max_value > beta:
                    break

            return max_value

        def get_min(gameState, curr_depth, agent, alpha, beta):

            if (gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)

            min_value = inf
            for action in gameState.getLegalActions(agent):

                if agent == gameState.getNumAgents() - 1:
                    current = get_max(gameState.generateSuccessor(agent, action), curr_depth, alpha, beta)
                else:
                    current = get_min(gameState.generateSuccessor(agent, action), curr_depth, agent + 1, alpha, beta)

                if min_value > current:
                    min_value = current

                if min_value < alpha:
                    break

                if min_value < beta:
                    beta = min_value

            return min_value

        alpha = -inf
        max_value = -inf
        best_action = None

        for action in gameState.getLegalActions(0):

            if alpha < max_value:
                alpha = max_value

            current = get_min(gameState.generateSuccessor(0, action), self.depth - 1, 1, alpha, inf)

            if current > max_value:
                max_value = current
                best_action = action

            if best_action is None:
                best_action = action

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def get_expectimax(gameState, curr_depth, agent):
            if (gameState.isWin() or gameState.isLose() or curr_depth == self.depth):
                return self.evaluationFunction(gameState)

            if agent == 0:
                max_value = -inf
                for action in gameState.getLegalActions(agent):
                    current = get_expectimax(gameState.generateSuccessor(agent, action), curr_depth, 1)

                    if (max_value < current):
                        max_value = current

                return max_value

            else:
                next_agent = agent + 1
                sum_values = 0

                if agent == gameState.getNumAgents() - 1:
                    for action in gameState.getLegalActions(agent):
                        value = get_expectimax(gameState.generateSuccessor(agent, action), curr_depth + 1, 0)
                        sum_values += value
                else:
                    for action in gameState.getLegalActions(agent):
                        value = get_expectimax(gameState.generateSuccessor(agent, action), curr_depth, next_agent)
                        sum_values += value

                return sum_values / len(gameState.getLegalActions(agent))

        max_value = -inf
        best_action = None

        for action in gameState.getLegalActions(0):
            current = get_expectimax(gameState.generateSuccessor(0, action), 0, 1)
            if current > max_value:
                max_value = current
                best_action = action

        return best_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()

    if (currentGameState.isWin() or currentGameState.isLose()):
        return scoreEvaluationFunction(currentGameState)

    #standardowa ocena - ilosc jedzenia na planszy
    score = scoreEvaluationFunction(currentGameState) - (5 * len(newFood.asList()))

    # duchy
    ghostPositions = []
    eatableGhostPositions = []
    for ghost in currentGameState.getGhostStates():
        dis = util.manhattanDistance(newPos, ghost.getPosition())
        if ghost.scaredTimer == 0:
            ghostPositions.append(dis)
        else:
            eatableGhostPositions.append(dis)

    # duchy zwykle
    minDist = 1
    if ghostPositions:
        minDist = inf
        for ghostPos in ghostPositions:
            if ghostPos < minDist:
                minDist = ghostPos

    score -= (1.0 / minDist)

    # duchy jadalne
    minDist = 0
    if eatableGhostPositions:
        minDist = inf
        for ghostPos in eatableGhostPositions:
            if ghostPos < minDist:
                minDist = ghostPos

    score -= minDist

    # jedzeinie najbliższe
    minDist = inf
    for food in newFood.asList():
        dist = util.manhattanDistance(food, newPos)
        if dist < minDist:
            minDist = dist

    score -= 2 * minDist
    return score


# Abbreviation
better = betterEvaluationFunction
