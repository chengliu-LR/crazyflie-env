import math
import logging
from gym.utils import seeding
import gym
import numpy as np
from crazyflie_env.envs.utils.action import ActionXY
from crazyflie_env.envs.utils.state import FullState, ObservableState
from crazyflie_env.envs.utils.util import get_ranger_reflection

class Robot():
    def __init__(self, config=None):
        self.radius = 0.1
        self.v_pref = 1.0 # max velocity that can choose
        self.time_step = None
        self.fov = 2 * np.pi
        self.num_rangers = 4
        self.max_ranger_dist = 3.0
        # TODO: theta can be modified, add to state
        self.theta_orientation = 0.0

        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.ranger_reflections = None

    def set_state(self, px, py, gx, gy, vx, vy, segments, radius=None):
        """Set initial state for the robot.
        Param: (position_x, position_y, goal_pos_x, goal_pos_y, vel_x, vel_y, obstacle_segments, [optional] radius).
        """
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.ranger_reflections = self.get_ranger_reflections(segments)

        if radius is not None:
            self.radius = radius

    def get_ranger_reflections(self, segments):
        """
        set ranger_reflections according to obstacles in the environment
        return np.array with shape (4,)
        """
        return get_ranger_reflection(segments=segments, fov= self.fov, n_reflections=self.num_rangers, max_dist=self.max_ranger_dist,
                                    xytheta_robot=np.hstack((self.get_position(), self.theta_orientation)))


    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.ranger_reflections)


    def get_observable_state(self):
        goal_distance = self.get_goal_distance()
        return ObservableState(goal_distance, self.vx, self.vy, self.ranger_reflections)


    def get_position(self):
        return np.array([self.px, self.py])


    def get_goal_position(self):
        return np.array([self.gx, self.gy])


    def get_velocity(self):
        return np.array([self.vx, self.vy])


    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]


    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    def compute_next_position(self, action, dt):
        """
        action: ActionXY
        """
        px = self.px + action.vx * dt
        py = self.py + action.vy * dt

        return np.array([px, py]) # implicitly return a tuple instead of two values


    def validate_action(self, action):
        assert isinstance(action, ActionXY)


    def step(self, action, segments):
        self.validate_action(action)
        next_position = self.compute_next_position(action, self.time_step)
        self.px, self.py = next_position[0], next_position[1]
        self.vx = action.vx
        self.vy = action.vy
        self.ranger_reflections = self.get_ranger_reflections(segments)
        
        return self.get_observable_state()
    

    def get_goal_distance(self):
        return np.linalg.norm(self.get_position() - self.get_goal_position())
    

    def reached_destination(self):
        return self.get_goal_distance() < self.radius