import math
import logging
from gym.utils import seeding
import gym
import numpy as np
from crazyflie_env.envs.utils.action import ActionXY
from crazyflie_env.envs.utils.state import ObservableState, FullState


class Robot():
    def __init__(self, config=None):
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.radius = 0.1
        self.v_pref = 1.0 # max velocity that can choose
        self.time_step = None


    def set_state(self, px, py, gx, gy, vx, vy, radius=None):
        """Set initial state for the robot.
        Param: (position_x, position_y, goal_pos_x, goal_pos_y, vel_x, vel_y, [optional] radius).
        """
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        if radius is not None:
            self.radius = radius


    def get_full_state(self):
        # TODO: add ranger sensor to state
        
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy)
    

    def get_next_full_state(self, action):
        self.validate_action(action)
        next_px, next_py = self.compute_next_position(action, self.time_step)
        next_vx = action.vx
        next_vy = action.vy
        return FullState(next_px, next_px, next_vx, next_vy, self.radius, self.gx, self.gy)


    def get_position(self):
        return self.px, self.py


    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]


    def get_goal_position(self):
        return self.gx, self.gy


    def get_velocity(self):
        return self.vx, self.vy


    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    def compute_next_position(self, action, dt):
        """
        action: ActionXY
        """
        px = self.px + action.vx * dt
        py = self.py + action.vy * dt
        return px, py


    def validate_action(self, action):
        assert isinstance(action, ActionXY)


    def step(self, action):
        self.validate_action(action)
        self.px, self.py = self.compute_next_position(action, self.time_step)
        self.vx = action.vx
        self.vy = action.vy
        return self.get_full_state()


    def reached_destination(self):
        return np.linalg.norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius