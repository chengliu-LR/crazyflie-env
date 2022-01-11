import gym
import math
import logging
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt
from matplotlib import animation
from crazyflie_env.envs.utils.collision import point_to_segment_dist
from crazyflie_env.envs.utils.state import ObservableState
from crazyflie_env.envs.utils.action import ActionXY
from crazyflie_env.envs.utils.robot import Robot
#plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

class CrazyflieEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Agent is controlled by a known and learnable policy.
        """
        self.time_limit = 25 # in seconds, not steps
        self.time_step = 0.2 # in seconds
        self.global_time = 0 # in seconds
        self.set_robot(Robot())

        # reward function
        self.success_reward = 1
        self.collision_penalty = -0.25
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = 0.5
    
        # simulation config
        self.square_width = 5.0 # width of the square environment
        self.goal_distance = 4.0 # initial distance to goal pos
        self.UP_RIGHT = (self.square_width, self.square_width)
        self.BOTTOM_RIGHT = (self.square_width, -self.square_width)
        self.BOTTOM_LEFT = (-self.square_width, -self.square_width)
        self.UP_LEFT = (-self.square_width, self.square_width)

        self.front_wall = (self.UP_LEFT, self.UP_RIGHT)
        self.right_wall = (self.UP_RIGHT, self.BOTTOM_RIGHT)
        self.back_wall = (self.BOTTOM_RIGHT, self.BOTTOM_LEFT)
        self.left_wall = (self.BOTTOM_LEFT, self.UP_LEFT)

        self.obstacles = [self.front_wall, self.right_wall, self.back_wall, self.left_wall]

        # visualization
        self.states = None
        self.action_values = None

    def set_robot(self, robot):
        self.robot = robot
        self.robot.time_step = self.time_step


    def reset(self):
        """
        Set robot at the center of the environment with zero initial velocity.
        Return: ObservableState(px, py, vx, vy, radius)
        """
        if self.robot is None:
            raise AttributeError('Robot has to be set!')
        self.robot.set_state(0, -self.goal_distance, 0, self.goal_distance, 0, 0) # set initial pos and vel
        self.states = list()
        ob = self.robot.get_full_state()

        return ob
    

    def step(self, action, update=True):
        """
        Compute action for robot, detect collision, update environment.
        Return: (ob, reward, done, info)
        """

        # collision detection
        # can use 353 in crowd_sim.py to detect collisions between the robot and a cylinder obstacle.
        dist_min = float('inf')
        collision = False
        next_position = np.array(self.robot.compute_next_position(action, self.time_step))
        for i, obstacle in enumerate(self.obstacles):
            closet_dist = point_to_segment_dist(obstacle[0], obstacle[1], next_position) - self.robot.radius
        
            if closet_dist < 0:
                collision = True
                break
            elif closet_dist < dist_min:
                dist_min = closet_dist

        # check if reaching the goal
        goal_reached = np.linalg.norm(next_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time > self.time_limit:
            reward = 0
            done = True
            info = "Timeout"
        elif collision:
            reward = self.collision_penalty
            done = True
            info = "Collision"
        elif goal_reached:
            reward = self.success_reward
            done = True
            info = "Goal reached"
        elif dist_min < self.discomfort_dist:
            reward = (dist_min - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = "Danger"
        else:
            reward = 0
            done = False
            info = "Nothing"
        
        if update:
            # store for visualization
            self.states.append(self.robot.get_full_state())
            # update agents
            self.robot.step(action)
            self.global_time += self.time_step

            ob = self.robot.get_full_state()
        
        return ob, reward, done, info


    def render(self, mode="video", output_file=None):
        pass