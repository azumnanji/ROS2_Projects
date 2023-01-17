import numpy as np
import matplotlib.pyplot as plt
import enum

DISTANCE = 0.05

class DIRECTION(enum.Enum):
    UP=(-1,0,DISTANCE)
    UP_RIGHT=(-1,1,np.sqrt(2)*DISTANCE)
    UP_LEFT=(-1,-1,np.sqrt(2)*DISTANCE)
    DOWN=(1,0,DISTANCE)
    DOWN_RIGHT=(1,1,np.sqrt(2)*DISTANCE)
    DOWN_LEFT=(1,-1,np.sqrt(2)*DISTANCE)
    RIGHT=(0,1,DISTANCE)
    LEFT=(0,-1,DISTANCE)


class node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def euclidean_dist(curr_x, curr_y, end_x, end_y):
    '''
    Calculate Manhattan Distance for heuristic
    Inputs
        curr_x: current bot x position
        curr_y: current bot y position
        end_x: user end x position
        end_y: user end y position
    Output
        dist: Manhattan distance
    '''
    dist = (curr_x - end_x)**2 + (curr_y - end_y)**2
    return dist


def astar(maze, start, end, cost_threshold=60):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    assert isinstance(start[0], int) and isinstance(start[1], int) and isinstance(end[0], int) and isinstance(end[1], int), \
        "Need to pass tuple ints"
    print(len(maze), len(maze[0]))
    # Create start and end node
    start_node = node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)
    # Loop until you find the end node
    while len(open_list) > 0:
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # print(current_node.position)
        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal, you can also implement what should happen if there is no possible path
        if current_node == end_node:
            # Complete here code to return the shortest path found
            # recursively add parent of previous node to path
            parent = current_node.parent
            path = [current_node.position]
            while parent:
                path.append(parent.position)
                parent = parent.parent

            return list(reversed(path))

        # Complete here code to generate children, which are the neighboring nodes. You should use 4 or 8 points connectivity for a grid.
        # children are up, down, left, right (or for 8 diagonal)
        for e in DIRECTION:
            row = current_node.position[0] + e.value[0]
            col = current_node.position[1] + e.value[1]
            child = node(current_node, (row,col))
            if row >= 0 and row < len(maze) and col >= 0 and col < len(maze[0]) and maze[row][col] < cost_threshold and child not in closed_list: 
                # if not an obstacle, calculate f = h + g
                child.g = e.value[2] + current_node.g + maze[row][col]
                child.h = euclidean_dist(row, col, end[0], end[1])
                child.f = child.h + child.g

                # check if child is in open list (only replace if cost is lower
                try:
                    index_child = open_list.index(child)
                    if child.f < open_list[index_child].f:
                        open_list[index_child] = child
                except ValueError:
                    open_list.append(child)
        
    # could not find end node
    return None
            

def main():

    # Load your maze here
    maze = np.load('cost_map.npy')
    maze = np.flip(maze, 0)
    
    # This is an example maze you can use for testing, replace it with loading the actual map
    # maze = [[0,   0,   0,   0,   1,   0, 0, 0, 0, 0],
    #         [0, 0.8,   1,   0,   1,   0, 0, 0, 0, 0],
    #         [0, 0.9,   1,   0,   1,   0, 1, 0, 0, 0],
    #         [0,   1,   0,   0,   1,   0, 1, 0, 0, 0],
    #         [0,   1,   0,   0,   1,   0, 0, 0, 0, 0],
    #         [0,   0,   0, 0.9,   0,   1, 0, 0, 0, 0],
    #         [0,   0, 0.9,   1,   1, 0.7, 0, 0, 0, 0],
    #         [0,   0,   0,   1,   0,   0, 0, 0, 0, 0],
    #         [0,   0,   0,   0, 0.9,   0, 0, 0, 0, 0],
    #         [0,   0,   0,   0,   0,   0, 0, 0, 0, 0]]

    
    
    # Define here your start and end points
    start = (92, 214)
    end = (41, 175)
    
    # Compute the path with your implementation of Astar
    path = np.asarray( astar(maze, start, end, 60), dtype=np.float)
    maze_plot=np.transpose(np.nonzero(maze))

    plt.plot(maze_plot[:,1], maze_plot[:,0], 'o')
    
    if not np.any(path): # If path is empty, will be NaN, check if path is NaN
        print("No path found")
    else:
        plt.plot(path[:,1], path[:,0])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()