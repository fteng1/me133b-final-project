#!/usr/bin/env python3
#
#   fp_localize.py
#
#   Code framework to localize a robot in a grid...
#
#   Places to edit are marked as TODO.
#
import numpy as np

from fp_utilities import Visualization, Robot, Node
from a_star import runAStar
import random 
import time 

random.seed(1)

#
#  Define the Walls 
# 
w = ['xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
     'x                   x           x               x',
     'x                   x           x               x',
     'x                   x           x               x',
     'x                   x           x               x',
     'x         xxxxxxxxxxx           x               x',
     'x         x         x                           x',
     'x         x                     xxxxxxxxxxxxxxxxx',
     'x         x                           x         x',
     'x         x                           x         x',
     'x         x           x               x         x',
     'x         x           x               x         x',
     'x                     x               x         x',
     'x                     x               x         x',
     'xxxxxxxxxxx                           x         x',
     'x                                               x',
     'x              xxx   xxx                        x',
     'x                                               x',
     'x               x     x          xxxxxxxxxxxxxxxx',
     'x               x     x                         x',
     'x               x     x                         x',
     'x                   xxx                         x',
     'x           xxxxxxxxxxxxxxxx                    x',
     'x                                               x',
     'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx']

walls = np.array([[1.0*(c == 'x') for c in s] for s in w])
rows  = np.size(walls, axis=0)
cols  = np.size(walls, axis=1)
startHider = (1, 45)
startSeeker = (12, 10)
goal  = (21, 17)
hiderBot = Robot(np.zeros((rows, cols)), row=startHider[0], col=startHider[1], goalRow = goal[0], goalCol = goal[1])
seekerBot = Robot(np.zeros((rows, cols)), row=startSeeker[0], col=startSeeker[1], goalRow = startSeeker[0], goalCol = startSeeker[1])
hiderProb = np.zeros((rows, cols))
probDecrement = 0.05
hiderMove = True 

def findPath(robot):
    # bfs
    start = (robot.row, robot.col) 
    goal = (robot.goalRow, robot.goalCol)
    queue = [Node(start[0], start[1])] 
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited = [start[0], start[1]]
    while len(queue) != 0: 
        curr = queue.pop(0) 
        if curr.coordinates() == goal: 
            break 
        for d in dirs: 
            x = curr.x + d[0]
            y = curr.y + d[1]
            nextNode = Node(x, y)
            nextNode.parent = curr
            if x < 0 or y < 0 or x >= rows or y >= cols or robot.found[x][y] == -1 or (x, y) in visited: 
                continue 
            visited.append((x, y))
            queue.append(nextNode)
    path = [] 
    while curr.parent is not None: 
        path.append(curr)
        curr = curr.parent 
    path.append(Node(start[0], start[1]))
    return path

def findInd(array, center, radius):
    a = np.indices(array.shape).reshape(2, -1).T
    ans = a[np.abs(a - np.array(center)).sum(1) <= int(radius)] - np.array(center)
    return ans.tolist()

def normalizeDir(dir): 
    x = 0 
    y = 0 
    if dir[0] != 0: 
        x = dir[0] / abs(dir[0])
    if dir[1] != 0: 
        y = dir[1] / abs(dir[1])
    return [x, y]

def filterOutWalls(robot, dirs, center):
    toRemove = []
    for d in dirs: 
        x = center[0] + d[0]
        y = center[1] + d[1]
        if walls[x][y] == 1:
            robot.found[x][y] = -1
            if d not in toRemove:
                toRemove.append(d)
            normD = normalizeDir(d)
            for dir in dirs: 
                dispWall = normalizeDir([dir[0] - d[0], dir[1] - d[1]])
                if normD[0] * dispWall[0] + normD[1] * dispWall[1] > 0: 
                    if dir not in toRemove:
                        toRemove.append(dir)
    for r in toRemove:
        dirs.remove(r)
                
    

def goPath(robot, visual, isSeeker):
    if isSeeker: 
        setNewSeekerGoal(robot)
    path = runAStar(rows, cols, robot.Position(), [robot.goalRow, robot.goalCol], robot.found)[::-1] 
    robot.currPath = path 
    if not isSeeker:
        radius = 3
        trace = findInd(walls, robot.Position(), radius)
        filterOutWalls(robot, trace, robot.Position())
        for t in trace:
            dist = np.sqrt(t[0] ** 2 + t[1] ** 2)
            if dist == 0: 
                hiderProb[robot.row + t[0]][robot.col + t[1]] += 1
            else:
                hiderProb[robot.row + t[0]][robot.col + t[1]] += 1 - dist/radius
        for r in range(rows): 
            for c in range(cols): 
                hiderProb[r][c] = max(0, hiderProb[r][c] - probDecrement)
    sensorRange = 1 
    if isSeeker: 
        sensorRange = 5
    dirs = findInd(walls, robot.Position(), sensorRange)
    filterOutWalls(robot, dirs, robot.Position())
 
    for d in dirs: 
        if isSeeker: 
            hiderRow = hiderBot.row 
            hiderCol = hiderBot.col
        x = robot.row + d[0]#  * i
        y = robot.col + d[1]# * i
    
        if x < 0 or y < 0 or x >= rows or y >= cols:
            continue 
        else: 
            if walls[x][y] == 1: 
                robot.found[x][y] = -1 
            if isSeeker and walls[x][y] != 1:
                robot.found[x][y] = hiderProb[x][y]
                if not hiderMove:
                    robot.found[x][y] = -10 
                if hiderRow == x and hiderCol == y:
                    print("found!")
                    return True

    # check if there is wall 
    if len(robot.currPath) > 0: 
        p = robot.currPath.pop(0)
        if robot.found[p.x][p.y] == -1:
            return False
        else: 
            # move robot
            drow = p.x - robot.row
            dcol = p.y - robot.col
            
            visual.Show(walls, goal, hiderBot.Position(), seekerBot.Position())
            robot.Command(drow, dcol)
    return False

def setNewSeekerGoal(robot):
    if robot.Position() == (robot.goalRow, robot.goalCol): 
        robot.found[robot.goalRow][robot.goalCol] -= 2 * probDecrement 
    maxIdx = np.argwhere(robot.found == robot.found.max())
    if robot.found.max() != 0 or robot.Position() == (robot.goalRow, robot.goalCol) or robot.found[robot.goalRow][robot.goalCol] == -1:
        if maxIdx.shape[0] == 1: 
            robot.goalRow = maxIdx[0][0]
            robot.goalCol = maxIdx[0][1]
            return

        if maxIdx.shape[0] > 1:
            rnd = np.random.randint(0, maxIdx.shape[0])
            maxIdx = maxIdx[rnd]
        robot.goalRow = maxIdx[0]
        robot.goalCol = maxIdx[1]


# 
#
#  Main Code
#
def main():
    # Initialize the figure.
    visual = Visualization(walls)

    start = time.time() 
    while hiderBot.Position() != (hiderBot.goalRow, hiderBot.goalCol): 
        if hiderMove:
            goPath(hiderBot, visual, False)
        if goPath(seekerBot, visual, True): 
            break
    print("Time: " + str(time.time() - start))
    bel = 1.0 - walls 
    bel = (1.0/np.sum(bel)) * bel

    # Pick the algorithm assumptions:
    probCmd      = 1.0                  
    probProximal = [1.0]                

    # Report.
    print("Localization is assuming probCmd = " + str(probCmd) +
          " and probProximal = " + str(probProximal))


    # # Loop continually.
    while True:
        # Show the current belief.  Also show the actual position.

        while True:
            key = input("Cmd (q=quit, i=up, m=down, j=left, k=right) ?")
            if   (key == 'q'):  return
            elif (key == 'i'):  (drow, dcol) = (-1,  0) ; break
            elif (key == 'm'):  (drow, dcol) = ( 1,  0) ; break
            elif (key == 'j'):  (drow, dcol) = ( 0, -1) ; break
            elif (key == 'k'):  (drow, dcol) = ( 0,  1) ; break



if __name__== "__main__":
    main()