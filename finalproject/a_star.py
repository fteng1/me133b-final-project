import bisect
import numpy as np
import matplotlib.pyplot as plt
from fp_utilities import Node

class State:
    # Possible status of each state.
    WALL      = -1      # Not a legal state - just to indicate the wall
    UNKNOWN   =  0      # "Air"
    ONDECK    =  1      # "Leaf"
    PROCESSED =  2      # "Trunk"
    PATH      =  3      # Processed and later marked as on path to goal

    STATUSSTRING = {WALL:      'WALL',
                    UNKNOWN:   'UNKNOWN',
                    ONDECK:    'ONDECK',
                    PROCESSED: 'PROCESSED',
                    PATH:      'PATH'}

    STATUSCOLOR = {WALL:      np.array([0.0, 0.0, 0.0]),   # Black
                   UNKNOWN:   np.array([1.0, 1.0, 1.0]),   # White
                   ONDECK:    np.array([0.0, 1.0, 0.0]),   # Green
                   PROCESSED: np.array([0.0, 0.0, 1.0]),   # Blue
                   PATH:      np.array([1.0, 0.0, 0.0])}   # Red

    # Initialization
    def __init__(self, row, col):
        # Save the location.
        self.row = row
        self.col = col

        # Clear the status and costs.
        self.status = State.UNKNOWN
        self.creach = 0.0       # Actual cost to reach
        self.cost   = 0.0       # Estimated total path cost (to sort)

        # Clear the references.
        self.parent    = None
        self.neighbors = []


    # Define less-than, so we can sort the states by cost.
    def __lt__(self, other):
        return self.cost < other.cost

    # Define the Manhattan distance.
    def distance(self, other):
        return abs(self.row - other.row) + abs(self.col - other.col)


    # Return the color matching the status.
    def color(self):
        return State.STATUSCOLOR[self.status]

    # Return the representation.
    def __repr__(self):
        return ("<State %d,%d = %s, cost %f>\n" %
                (self.row, self.col,
                 State.STATUSSTRING[self.status], self.cost))



#
#   A* Algorithm
#
# Estimate the cost to go from state to goal.
def costtogo(state, goal):
    # Change c to 0 for part a, 1 for part b, 2 for part c, and 10 for part d
    c = 10
    return c * state.distance(goal)

# Run the full A* algorithm.
def astar(start, goal):
    # Prepare the still empty *sorted* on-deck queue.
    onDeck = []

    # Setup the start state/cost to initialize the algorithm.
    start.status = State.ONDECK
    start.creach = 0.0
    start.cost   = costtogo(start, goal)
    start.parent = None
    bisect.insort(onDeck, start)

    while True:
        nextState = onDeck.pop(0) 
        # Loop through all possible neighbors
        for neighbor in nextState.neighbors: 
            # Neighbor has not been processed previously
            if neighbor.status == State.UNKNOWN: 
                # Add neighbor to on deck 
                neighbor.status = State.ONDECK 
                neighbor.creach = nextState.creach + 1.0
                neighbor.cost = neighbor.creach + costtogo(neighbor, goal) 
                neighbor.parent = nextState 
                bisect.insort(onDeck, neighbor) 
            # Neighbor is already on deck
            elif neighbor.status == State.ONDECK: 
                # Only change cost if new shortest path is found
                if nextState.creach + 1.0 < neighbor.creach: 
                    # Update with new cost and sort on deck list accordingly
                    onDeck.remove(neighbor)
                    neighbor.creach = nextState.creach + 1.0 
                    neighbor.cost = neighbor.creach + costtogo(neighbor, goal)
                    neighbor.parent = nextState 
                    bisect.insort(onDeck,neighbor)
            else: 
                # If the neighbor is done, ignore state 
                pass 
        nextState.status = State.PROCESSED 
        # Exit loop if we reach goal or have searched all reachable states
        if nextState == goal or len(onDeck) == 0: 
            break


    # Check if path was found to goal 
    currentNode = goal 
    ans = []
    while currentNode.parent != None:
        ans.append(Node(currentNode.row, currentNode.col))
        currentNode.status = State.PATH 
        currentNode = currentNode.parent 
    if currentNode == start: 
        start.status = State.PATH

    return ans

def runAStar(rows, cols, start, goal, walls):
    # Create a grid of states.
    M = rows
    N = cols
    states = [[State(m,n) for n in range(N)] for m in range(M)]

    for (m,n) in np.argwhere(walls == -1.0):
        states[m][n].status = State.WALL

    # Set the neighbors - this makes sure the full graph is implemented.
    for m in range(M):
        for n in range(N):
            if not states[m][n].status == State.WALL:
                for (m1, n1) in [(m-1,n), (m+1,n), (m,n-1), (m,n+1)]:
                    if m1 >= 0 and m1 < rows and n1 >= 0 and n1 < cols:
                        if not states[m1][n1].status == State.WALL:
                            states[m][n].neighbors.append(states[m1][n1])

    # Pick the start/goal states.
    start = states[start[0]][start[1]]
    goal  = states[goal[0]][goal[1]]

    # Run the A* algorithm.
    return astar(start, goal)