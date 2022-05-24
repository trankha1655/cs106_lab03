"""#
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()



def aStarSearch(wrapper):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """Implement A start approach"""

    start_state, init_cost = wrapper.get_start_state()
    frontier = PriorityQueue()
    frontier.push([start_state], init_cost)                                          # Init state_stack with inital state and 0 cost

    exploredSet = set()                                     
    actions = PriorityQueue()
    actions.push([0], init_cost)                                                    # Init action_stack with inital no action and 0 cost                                  
    temp = []
    ### Implement uniform cost search here

    
    while frontier:
        if frontier.isEmpty():                                               # If the frontier is empty, it means there is no way to solve the game (box adjacent to the wall)
            return []
        _, node = frontier.pop()                                             # Get the sequence of state having the lowest cost
        action_cost, node_action = actions.pop()                             # Get the sequence of action having the lowest cost

        if wrapper.is_end(node[-1]):                                           # Verify last action to check whether this sequence of action is end
            temp += node_action[1:]                                            # Get all actions except the first 1
            break
        
        if node[-1] not in exploredSet:                                           # Check the last state in sequence is explored or not 
            exploredSet.add(node[-1])                                             # if no, mark this state is explored
            for action in wrapper.get_all_legal_actions(node[-1]):                 # Loop through all available action in the last state
                next_state, _ = wrapper.get_successors(node[-1], action)           # Update state (playing that action)

                new_cost = len(node_action) + wrapper.get_mht_distance(next_state)   # we asume that every action having cost = 1
                frontier.push(node + [next_state], new_cost)                    # add old state + new explored state and cost of the state to frontier
                actions.push(node_action + [action], new_cost)                  # add old action + new explored action and cost to action_stack

    return temp
    


# Abbreviations
astar = aStarSearch

