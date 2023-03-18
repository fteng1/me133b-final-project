#!/usr/bin/env python3
#
#   hw5_utilities.py
#
#   Two classes to help the visualization and simulation.
#
#
#   VISUALIZATION:
#
#     visual = Visualization(found)
#     visual.Show(prob)
#     visual.Show(prob, (row, col))
#
#     found         NumPy 2D array defining both the grid size
#                   (rows/cols) and found (being non-zero elements).
#     prob          NumPy 2D array of probabilities, values 0 to 1.
#
#   visual.Show() will visualize the probabilities.  The second form
#   will also mark the (row,col) position with an 'x'.
#
#
#   ROBOT SIMULATION:
#
#     robot = Robot(found, row=0, col=0, probCmd=1.0, probProximal=[1.0])
#     robot.Command(drow, dcol)
#     True/False = robot.Sensor(drow, dcol)
#     (row, col) = robot.Position()
#
#     probCmd       Probability the command is executed (0 to 1).
#     probProximal  List of probabilities (0 to 1).  Each element is
#                   the probability that the proximity sensor will
#                   fire at a distance of (index+1 = 1,2,3,etc)
#     (drow, dcol)  Delta up/right/down/left: (-1,0) (0,1) (1,0) (0,-1)
#
#   Simulate a robot, to give us the sensor readings.  If the starting
#   row/col are not given, pick them randomly.  Note both the command
#   and the sensor may be configured to a random probability level.
#   The robot.Position() is intended for debugging and visualization
#   only, as it gives the actual position.
#
import matplotlib.pyplot as plt
import numpy as np
import random


#
#   Probailiity Grid Visualization
#
class Visualization():
    def __init__(self, found):
        # Save the found and determine the rows/cols:
        self.found = found
        self.spots = np.sum(np.logical_not(found))
        self.rows  = np.size(found, axis=0)
        self.cols  = np.size(found, axis=1)
        # Create the color range.  There are clearly more elegant ways...
        self.color = np.ones((self.rows, self.cols, 3))

        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(False)
        plt.gca().axis('off')
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim(0, self.cols)
        plt.gca().set_ylim(self.rows, 0)

        # Add the row/col numbers.
        for row in range(0, self.rows, 2):
            plt.gca().text(         -0.3, 0.5+row, '%d'%row,
                           verticalalignment='center',
                           horizontalalignment='right')
        for row in range(1, self.rows, 2):
            plt.gca().text(self.cols+0.3, 0.5+row, '%d'%row,
                           verticalalignment='center',
                           horizontalalignment='left')
        for col in range(0, self.cols, 2):
            plt.gca().text(0.5+col,          -0.3, '%d'%col,
                           verticalalignment='bottom',
                           horizontalalignment='center')
        for col in range(1, self.cols, 2):
            plt.gca().text(0.5+col, self.rows+0.3, '%d'%col,
                           verticalalignment='top',
                           horizontalalignment='center')

        # Draw the grid, zorder 1 means draw after zorder 0 elements.
        for row in range(self.rows+1):
            plt.gca().axhline(row, lw=1, color='k', zorder=1)
        for col in range(self.cols+1):
            plt.gca().axvline(col, lw=1, color='k', zorder=1)

        # Clear the content and mark.  Then show with zeros.
        self.content = None
        self.hMark   = None
        self.sMark   = None
        # self.Show(np.zeros((self.rows, self.cols)), )
    
    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x),
                 (head.y, tail.y), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)

    def Flush(self):
        # Show the plot.
        plt.pause(0.1)

    def Mark(self, hRow, hCol, sRow, sCol):
        # Check the row/col arguments.
        assert (hRow >= 0) and (hRow < self.rows), "Illegal row"
        assert (hCol >= 0) and (hCol < self.cols), "Illegal col"
        assert (sRow >= 0) and (sRow < self.rows), "Illegal row"
        assert (sCol >= 0) and (sCol < self.cols), "Illegal col"

        # Potentially remove the previous mark.
        if self.hMark is not None:
            self.hMark.remove()
            self.hMark = None
            
        if self.sMark is not None:
            self.sMark.remove()
            self.sMark = None

        # Draw the mark.
        self.hMark  = plt.gca().text(0.5+hCol, 0.5+hRow, 'h', color = 'green',
                                    verticalalignment='center',
                                    horizontalalignment='center',
                                    zorder=1)
        self.sMark  = plt.gca().text(0.5+sCol, 0.5+sRow, 's', color = 'red',
                                    verticalalignment='center',
                                    horizontalalignment='center',
                                    zorder=1)

    def Grid(self, prob, hiderGoal):
        # Check the probability grid array size.
        assert (np.size(prob, axis=0) == self.rows), "Inconsistent num of rows"
        assert (np.size(prob, axis=1) == self.cols), "Inconsistent num of cols"

        # Potentially remove the previous grid/content.
        if self.content is not None:
            self.content.remove()
            self.content = None


        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.found[row,col]:
                    self.color[row,col,0:3] = np.array([0.0, 0.0, 0.0])   # Black
                    self.color[hiderGoal[0], hiderGoal[1], 0:3] = np.array([0.0, 1.0, 1.0])
                else:
                    # Shades of pink/purple/blue. Yellow means impossible.
                    p    = prob[row,col]
                    pmin = 0.9 / self.spots
                    if p == 0:
                        self.color[row,col,0:3] = np.array([1.0, 1.0, 0.0])
                    elif p < pmin:
                        rlevel = (1.0 - p)
                        glevel = (1.0 - p)
                        self.color[row,col,0:3] = np.array([rlevel, glevel, 1.0])
                    else:
                        rlevel = (1.0 - p)
                        glevel = (pmin/p - pmin)
                        self.color[row,col,0:3] = np.array([rlevel, glevel, 1.0])
    
        # Draw the boxes.
        self.content = plt.gca().imshow(self.color,
                                        aspect='equal',
                                        interpolation='none',
                                        extent=[0, self.cols, self.rows, 0],
                                        zorder=0)
    def ShowSensorRange(self, dirs, centerx, centery):
        for d in dirs:
            x = centerx + d[0]
            y = centery + d[1]
            self.color[x,y,0:3] = np.array([0.0, 0.0, 1.0])
        self.content = plt.gca().imshow(self.color,
                                        aspect='equal',
                                        interpolation='none',
                                        extent=[0, self.cols, self.rows, 0],
                                        zorder=0) 
        self.Flush()
    
    def Show(self, prob, hiderGoal,  hiderPos = None, seekerPos = None):
        # Update the content.
        self.Grid(prob, hiderGoal)

        # Potentially add the mark.
        if hiderPos is not None and seekerPos is not None:
            self.Mark(hiderPos[0], hiderPos[1], seekerPos[0], seekerPos[1])

        # Flush the figure.
        self.Flush()

#
#  Robot (Emulate the actual robot)
#
#    probCmd is the probability that the command is actually executed
#
#    probProximal is a list of probabilities.  Each element
#    corresponds to the probability that the proximity sensor will
#    fire at a distance of (index+1).



class Robot():
    def __init__(self, found, row = 0, col = 0, goalRow = 0, goalCol = 0,
                 probCmd = 1.0, probProximal = [1.0], currPath = []):
        # Check the row/col arguments.
        assert (row >= 0) and (row < np.size(found, axis=0)), "Illegal row"
        assert (col >= 0) and (col < np.size(found, axis=1)), "Illegal col"

        # Report.
        if found[row, col]:
            location = " (random location)"
        else:
            location = " (at %d, %d)" % (row, col)
        print("Starting robot with real probCmd = " + str(probCmd) +
              " and probProximal = " + str(probProximal) + location)

        # Save the found, the initial location, and the probabilities.
        self.found        = found
        self.row          = row
        self.col          = col
        self.probCmd      = probCmd
        self.probProximal = probProximal
        self.goalRow      = goalRow
        self.goalCol      = goalCol
        self.currPath     = currPath

        # Pick a valid starting location (if not already given).
        while self.found[self.row, self.col]:
            self.row = random.randrange(0, np.size(found, axis=0))
            self.col = random.randrange(0, np.size(found, axis=1))

    def Command(self, drow, dcol):
        # Check the delta.
        assert ((abs(drow+dcol) == 1) and (abs(drow-dcol) == 1)), "Bad delta"
        
        # Try to move the robot the given delta.
        row = self.row + drow
        col = self.col + dcol
        self.row = row
        self.col = col

    def Position(self):
        return (self.row, self.col)

    def Sensor(self, drow, dcol): 
        # Check the delta.
        assert ((abs(drow+dcol) == 1) and (abs(drow-dcol) == 1)), "Bad delta"

        # Check the proximity in the given direction.
        for k in range(len(self.probProximal)):
            if self.found[self.row + drow*(k+1), self.col + dcol*(k+1)]:
                return (random.random() < self.probProximal[k])
        return False
    
######################################################################
#
#   Node Definition
#
class Node:
    # Initialize with coordinates.
    def __init__(self, x, y):
        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y

        # Clear any parent information.
        self.parent = None

    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        return (self.x, self.y)
