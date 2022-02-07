import gym
import math
import logging
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.lines as matlines
from matplotlib import animation
from matplotlib import patches
from crazyflie_env.envs.utils.util import point_to_segment_dist
from crazyflie_env.envs.utils.state import ObservableState, FullState
from crazyflie_env.envs.utils.action import ActionXY
from crazyflie_env.envs.utils.robot import Robot
#plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

class CrazyflieEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """Agent is controlled by a known and learnable policy.
        """
        self.time_limit = 25 # in seconds, not steps
        self.time_step = 0.2 # in seconds
        self.global_time = 0 # in seconds
        self.set_robot(Robot())

        # reward function
        self.success_reward = 2
        self.collision_penalty = -2.5
        self.goal_distance_penalty_factor = -0.1
        self.discomfort_dist = 0.5
        self.discomfort_penalty_factor = 1
        self.speed_penalty_factor = -0.5
    
        # simulation config
        self.square_width = 5.0 # width of the square environment
        self.goal_distance = 3.0 # initial distance to goal pos
        self.UP_RIGHT = (self.square_width, self.square_width)
        self.BOTTOM_RIGHT = (self.square_width, -self.square_width)
        self.BOTTOM_LEFT = (-self.square_width, -self.square_width)
        self.UP_LEFT = (-self.square_width, self.square_width)

        self.front_wall = (self.UP_LEFT, self.UP_RIGHT)
        self.right_wall = (self.UP_RIGHT, self.BOTTOM_RIGHT)
        self.back_wall = (self.BOTTOM_RIGHT, self.BOTTOM_LEFT)
        self.left_wall = (self.BOTTOM_LEFT, self.UP_LEFT)

        self.obstacles = [self.front_wall, self.right_wall, self.back_wall, self.left_wall]
        #self.added_walls = [((0, -2), (-5, -2)), ((0, 2), (5, 2))]
        #self.added_walls = [((0, -2), (-5, -2))]
        self.added_walls = []

        # visualization
        self.states = None
        self.action_values = None

    def set_robot(self, robot):
        self.robot = robot
        self.robot.time_step = self.time_step


    def reset(self):
        """Set robot at the center of the environment with zero initial velocity.
        Return: ObservableState(px, py, vx, vy, radius)
        """
        self.global_time = 0
        if self.robot is None:
            raise AttributeError('Robot has to be set!')
        self.robot.set_state(0, -self.goal_distance, 0, self.goal_distance, 0, 0) # set initial pos and vel
        self.states = list()
        ob = self.robot.get_full_state()

        return ob
    

    def step(self, action, update=True):
        """Compute action for robot, detect collision, update environment.
        Return: (ob, reward, done, info)
        """

        # collision detection
        # can use 353 in crowd_sim.py to detect collisions between the robot and a cylinder obstacle.
        dist_min = float('inf')
        collision = False
        next_position = np.array(self.robot.compute_next_position(action, self.time_step))
        for i, obstacle in enumerate(self.obstacles + self.added_walls):
            closet_dist = point_to_segment_dist(obstacle[0], obstacle[1], next_position) - self.robot.radius
        
            if closet_dist < 0:
                collision = True
                break
            elif closet_dist < dist_min:
                dist_min = closet_dist

        # check if reaching the goal
        goal_distance = np.linalg.norm(next_position - np.array(self.robot.get_goal_position()))
        goal_reached = goal_distance < self.robot.radius

        reward = self.goal_distance_penalty_factor * goal_distance

        speed_limit_exceeded = bool(np.abs(action.vx) > 0.5 or np.abs(action.vy) > 0.5)
        if speed_limit_exceeded:
            reward += self.speed_penalty_factor * np.random.binomial(1, 0.4, 1)[0]

        if self.global_time > self.time_limit:
            #reward = 0
            done = True
            info = "Timeout"
        elif collision:
            reward += self.collision_penalty
            done = True
            info = "Collision"
        elif goal_reached:
            reward += self.success_reward
            done = True
            info = "Goal Reached"
        elif dist_min < self.discomfort_dist:
            reward += (dist_min - self.discomfort_dist) * self.discomfort_penalty_factor
            done = False
            info = "Danger"
        else:
            #reward = self.goal_distance_penalty_factor * goal_distance
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


    def render(self, mode='video', output_file=None):
        #plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'aquamarine'
        goal_color = 'red'
        arrow_color = 'orange'
        arrow_style = patches.ArrowStyle("simple", head_length=5, head_width=3)

        if mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            
            # add robot and its goal
            robot_positions = [state.position for state in self.states]
            goal = matlines.Line2D(xdata=[0], ydata=[3], color=goal_color, marker="*", linestyle='None', markersize=20, label='Goal')

            # add walls if any
            #wall1 = matlines.Line2D(xdata=[0,-5], ydata=[-2,-2], color='grey')
            #ax.add_artist(wall1)
            #wall2 = matlines.Line2D(xdata=[0,5], ydata=[1,1], color='grey')

            robot_ = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot_)
            ax.add_artist(goal)
            #ax.add_artist(wall2)
            plt.legend([robot_, goal], ['Robot', 'Goal'], fontsize=16)

            # add time annotation
            time_annotation = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time_annotation)
            # compute orientation, use arrow to show the direction
            radius = self.robot.radius
            orientation = []
            for state in self.states:
                theta = np.arctan2(state.vy, state.vx)
                orientation.append(((state.px, state.py), (state.px + radius * np.cos(theta), state.py + radius * np.sin(theta))))

            # at time zero
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)]

            for arrow in arrows: # only a robot in this case
                ax.add_artist(arrow)
            global_step = 0
            
            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot_.center = robot_positions[frame_num]
                for arrow in arrows: # only a robot in this
                    arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color, arrowstyle=arrow_style)]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                time_annotation.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
            
            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me', bitrate=1800))
                anim.save(output_file, writer=writer)
            else:
                plt.show()
