from typing import Dict, List, Tuple
import numpy as np


class GridWorld:
    """
    This class represents environment, where actions take place.
    """
    class Object:
        """
        Class for object that contain mask - set of coordinates from pivot position, later it would help to understand
        whether object might be placed in some other positions.
        Pivot position - left-lower position in the smallest rectangle that might cover object.
        """

        def __init__(self, object_positions):
            # https://stackoverflow.com/questions/14802128/tuple-pairs-finding-minimum-using-python
            self.pivot = (max(object_positions, key=lambda t: t[0])[0], max(object_positions, key=lambda t: t[1])[1])
            self.mask = []
            for x, y in object_positions:
                self.mask.append((x - self.pivot[0], y - self.pivot[1]))

    def is_position_available(self, target_position):
        """
        This function with help of mask of object checks, if object might be placed with pivot in given position.
        :param target_position: position which is considered during generate_action() function.
        :return: Bool value that determines if there is a possibility to place object in required position.
        """
        for mask_x, mask_y in self.object.mask:
            x = target_position[0] + mask_x
            y = target_position[1] + mask_y

            if x < 0 or x >= self.height or y < 0 or y >= self.width or (x, y) in self.obstacles:
                return False

        return True

    def generate_actions(self):
        """
        This function adds in dictionary all possible actions for each position for object in environment.
        :return:
        """
        for x, y in self.positions:
            if (x, y) == self.final_position or not self.is_position_available((x, y)):
                continue

            target_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

            for target_position in target_positions:
                if self.is_position_available(target_position):
                    self.actions[(x, y)].append(target_position)

    def compute_value_for_position(self, position):
        """
        This function calculate value function for certain position and determines in which position is better to go
        in order to maximize value function (let's call it neighbour position) if there is such possible move. In other
        words it is position in which move from current position is whether the only possible or which maximizes
        value function.
        :param position: requested position
        :return: value for value function and best neighbour position, if there is no such returns itself.
        """
        if len(self.actions[position]) == 0:
            return self.value_function[position], position

        max_val = float("-inf")
        best_neighbour = position

        for place in self.actions[position]:
            action_reward = self.rewards.get((position, place), self.default_reward)
            next_v = self.gamma * self.value_function[place]

            if action_reward + next_v > max_val:
                max_val = action_reward + next_v
                best_neighbour = place

        return max_val, best_neighbour

    def update_policy(self, position, best_neighbour):
        """
        This function adds in policy dictionary tuple of best neighbour position and label, which shows direction of the
        move.
        :param position: requested position
        :param best_neighbour: position in which move is whether the only possible or which maximizes value function.
        :return:
        """
        x = best_neighbour[0] - position[0]
        y = best_neighbour[1] - position[1]

        label = '-'

        if x > 0:
            label = 'D'
        if y > 0:
            label = 'R'
        if x < 0:
            label = 'U'
        if y < 0:
            label = 'L'

        self.policy[position] = (best_neighbour, label)

    def update_values(self):
        """
        This function updates value function for all positions.
        :return: maximum change in value function
        """
        new_value_function = self.value_function.copy()
        max_change = float("-inf")

        for position in self.positions:
            new_value_function[position], best_neighbour = self.compute_value_for_position(position)
            self.update_policy(position, best_neighbour)
            temp_change = abs(new_value_function[position] - self.value_function[position])

            if temp_change > max_change:
                max_change = temp_change

        self.value_function = new_value_function
        return max_change

    def construct_way(self):
        """
        This function constructs the output which shows the sequence of moves which leads to desirable final position,
        if there is such possibility.
        :return: message for output.
        """
        if self.final_position == self.object.pivot:
            return "Object already in most lower-right position"

        sequence = ""
        current_position = self.object.pivot
        visited_positions = []

        while current_position != self.final_position and current_position not in visited_positions:
            sequence += self.policy[current_position][1] + " "
            visited_positions.append(current_position)
            current_position = self.policy[current_position][0]

        if self.final_position != current_position:
            return "No path"

        return sequence

    def Find_Da_Wae(self, iterations=100, delta: float = 0.001):
        """
        Uganda knuckles starts to calculate value function until the change in it would be lower that required precision
        or number of updates overcome number of given iterations. After calculations, they are starting to construction
        the way.
        :param iterations: Maximum number of possible updates.
        :param delta: Some precision.
        :return: Constructed way, if there is some.
        """
        term = 0
        dif = 1

        while term < iterations and dif > delta:
            dif = self.update_values()
            term += 1
        # self.visualize_value_function()
        return self.construct_way()

    def visualize_value_function(self):
        """
        This function just visualise environment.
        :return:
        """
        array = np.zeros((self.height, self.width))
        for x in range(self.height):
            for y in range(self.width):
                array[x, y] = self.value_function[(x, y)]
        print(array)

        array = np.empty((self.height, self.width), str)
        for x in range(self.height):
            for y in range(self.width):
                _, array[x, y] = self.policy[(x, y)]
        print(array)

    def __init__(self,
                 grid: List[List[int]],
                 rewards: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {},
                 gamma: float = 0.9,
                 default_reward=-1,
                 finish_value: float = 10,):

        self.gamma = gamma
        self.rewards = rewards
        self.default_reward = default_reward

        self.height = len(grid)
        self.width = len(grid[0])

        self.final_position = (self.height - 1, self.width - 1)
        self.value_function = {}
        self.obstacles = set()
        self.object_positions = []
        self.positions = []
        self.policy = {}
        self.actions = {}

        for x in range(self.height):
            for y in range(self.width):
                position = (x, y)
                self.positions.append(position)
                self.actions[position] = []
                self.value_function[position] = 0.0
                self.policy[position] = (position, "-")

                if grid[x][y] == 1:
                    self.obstacles.add(position)

                if grid[x][y] == 2:
                    self.object_positions.append(position)

        self.value_function[self.final_position] = finish_value

        self.object = self.Object(self.object_positions)
        self.generate_actions()


def find_path(path_to_infile, path_to_outfile):
    infile = open(path_to_infile, "r")

    grid = []
    for line in infile.readlines():
        grid.append([eval(element) for element in line.split()])
    infile.close()

    if not grid:
        print("File is empty")
        sys.exit(1)

    world = GridWorld(grid)
    answer = world.Find_Da_Wae()

    outfile = open(path_to_outfile, "w")
    outfile.write(answer)
    outfile.close()
