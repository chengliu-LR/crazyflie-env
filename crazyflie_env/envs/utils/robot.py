import math
import logging
from gym.utils import seeding
import gym
import numpy as np
from crazyflie_env.envs.utils.action import ActionXY, ActionRotation
from crazyflie_env.envs.utils.state import FullState, ObservableState
from crazyflie_env.envs.utils.util import get_ranger_reflection

class Robot():
    def __init__(self, config=None):
        self.radius = 0.1
        self.v_pref = 1.0 # max possible velocity
        self.time_step = None
        self.fov = 2 * np.pi
        self.num_rangers = 4
        self.max_ranger_dist = 3.0

        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vf = None
        self.orientation = None
        self.ranger_reflections = None


    def set_state(self, px, py, gx, gy, vf, orientation, segments, radius=None):
        """Set initial state for the robot.
        Param: (position_x, position_y, goal_pos_x, goal_pos_y, vel_forward, theta_orientation, obstacle_segments, [optional] radius).
        Param Orientation: in rads
        """
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vf = vf
        self.orientation = orientation
        self.ranger_reflections = self.get_ranger_reflections(segments)

        if radius is not None:
            self.radius = radius


    def get_ranger_reflections(self, segments):
        """
        set ranger_reflections according to obstacles in the environment
        return np.array with shape (4,)
        """
        return get_ranger_reflection(segments=segments, fov= self.fov, n_reflections=self.num_rangers, max_dist=self.max_ranger_dist,
                                    xytheta_robot=np.hstack((self.get_position(), self.orientation)))


    def get_full_state(self):
        return FullState(self.px, self.py, self.vf, self.radius, self.gx, self.gy, self.orientation, self.ranger_reflections)


    def get_observable_state(self):
        # TODO1: add orientation to observation
        # TODO2: historical ranger inputs
        return ObservableState(self.get_goal_distance(), self.orientation, self.ranger_reflections)


    def get_position(self):
        return np.array([self.px, self.py])


    def get_goal_position(self):
        return np.array([self.gx, self.gy])


    def compute_next_orientation(self, action, dt):
        """
        action: ActionRotation
        rot > 0: ccw
        rot < 0: cw
        """
        orientation = (self.orientation + action.rot * dt) % (2 * np.pi)

        return orientation

    def compute_next_position(self, orientation, action, dt):
        """
        compute next position according to forward velocity and orientation
        """
        vx = action.vf * np.cos(orientation)
        vy = action.vf * np.sin(orientation)

        px = self.px + vx * dt
        py = self.py + vy * dt

        return np.array([px, py]) # implicitly return a tuple instead of two values


    def validate_action(self, action):
        assert isinstance(action, ActionRotation)


    def step(self, action, segments):
        """
        update orientation, position, and ranger reflection to next state.
        note that orientation and position won't be updated at same step.
        """
        self.validate_action(action)
        next_orientation = self.compute_next_orientation(action, self.time_step)
        next_position = self.compute_next_position(next_orientation, action, self.time_step)
        self.px, self.py = next_position[0], next_position[1]
        self.vf = action.vf
        self.orientation = next_orientation
        self.ranger_reflections = self.get_ranger_reflections(segments)
        
        return self.get_observable_state()
    

    def get_goal_distance(self):
        return np.linalg.norm(self.get_position() - self.get_goal_position())
    

    def reached_destination(self):
        return self.get_goal_distance() < self.radius