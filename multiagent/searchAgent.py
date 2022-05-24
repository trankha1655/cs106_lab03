# searchAgents.py
# ---------------
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

"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Actions
from util import manhattanDistance
import search


ACTION = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

class PositionSearchProblem(search.SearchProblem):


    def __init__(self, game_state, start = None, goal = None, costFn = lambda x: 1):    
        self.walls = game_state.getWalls()
        self.start = start
        self.goal = goal
        self.costFn = costFn

    def get_start_state(self):
        return self.start, self.get_mht_distance(self.start)

    def is_end(self, state):
        return state == self.goal

    def get_mht_distance(self, state):
        return manhattanDistance(state, self.goal) 
    
    def get_all_legal_actions(self, state):
        x, y = state
        available_actions = []
        for action in ACTION:
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                available_actions.append(action)

        return available_actions

    def get_successors(self, state, action):
        x, y = state
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        return (nextx, nexty) , 1 #self.costFn(1)


def calculate_ghost_distance(game_state, ghost_id):
    pacman_position = game_state.getPacmanPosition()
    ghost_position = game_state.getGhostPosition(ghost_id)
    wrapper = PositionSearchProblem(game_state, start = pacman_position, goal = ghost_position)
    return len(search.astar(wrapper))


def calculate_maze_distance(game_state, destination):
    pacman_position = game_state.getPacmanPosition()
    # wrapper = GhostSearchWrapper(game_state, ghost_idx = ghost_id)
    wrapper = PositionSearchProblem(game_state, start = pacman_position, goal = destination)
    return len(search.astar(wrapper))