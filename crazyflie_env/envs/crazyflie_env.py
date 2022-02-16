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
from crazyflie_env.envs.utils.state import FullState
from crazyflie_env.envs.utils.action import ActionRotation
from crazyflie_env.envs.utils.robot import Robot
from crazyflie_env.envs.utils.obstacle import Obstacle
#plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

class CrazyflieEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """Agent is controlled by a known and learnable policy.
        """
        self.time_limit = 25 # in seconds, not steps
        self.time_step = 0.05 # in seconds
        self.global_time = 0 # in seconds
        self.random_init = True # randomly initialize robot position
        self.robot_po = False # partial observability of the robot
        self._set_robot(Robot(self.robot_po))

        # # reward function
        # self.success_reward = 5.0
        # self.collision_penalty = -5.0
        # self.goal_dist_reward_factor = 0.01
        # self.discomfort_dist = 0.2
        # self.discomfort_penalty_factor = -0.05
        # self.rotation_penalty_factor = -0.05

        # reward function
        self.success_reward = 50.0
        self.collision_penalty = -25.0
        self.goal_dist_penalty_factor = -2
        self.discomfort_dist = 0.5
        self.discomfort_penalty_factor = -5.0
        self.rotation_penalty_factor = -0.0

        # goal reaching radius
        self.goal_reaching_radius = 5 * self.robot.radius

        # simulation config
        self.square_width = 3.0 # half width of the square environment
        self.goal_height = 2.0 # initial distance to goal pos
        self.UP_RIGHT = (self.square_width, self.square_width)
        self.BOTTOM_RIGHT = (self.square_width, -self.square_width)
        self.BOTTOM_LEFT = (-self.square_width, -self.square_width)
        self.UP_LEFT = (-self.square_width, self.square_width)

        self.front_wall = (self.UP_LEFT[0], self.UP_LEFT[1], self.UP_RIGHT[0], self.UP_RIGHT[1])
        self.right_wall = (self.UP_RIGHT[0], self.UP_RIGHT[1], self.BOTTOM_RIGHT[0], self.BOTTOM_RIGHT[1])
        self.back_wall = (self.BOTTOM_RIGHT[0], self.BOTTOM_RIGHT[1], self.BOTTOM_LEFT[0], self.BOTTOM_LEFT[1])
        self.left_wall = (self.BOTTOM_LEFT[0], self.BOTTOM_LEFT[1], self.UP_LEFT[0], self.UP_LEFT[1])
        self.walls = [self.front_wall, self.right_wall, self.back_wall, self.left_wall] # ((x1, y1), (x1', y1'))

        self.obstacles = self.generate_obstacles(obstacle_num=0)
        self.obstacle_segments = self.walls + self.obstacles2segments(self.obstacles) # list of segments (x1, y1, x1', y1')

        # visualization, store full state of the robot
        self.states = None


    def _set_robot(self, robot):
        self.robot = robot
        self.robot.time_step = self.time_step


    def obstacles2segments(self, obstacles=[]):
        all_segments = []
        if obstacles is not None:
            for obstacle in obstacles:
                segments_each_obs = obstacle.get_segments()
                all_segments += segments_each_obs

        return all_segments


    def check_collision(self, position):
        """
        position: np.array, current robot position
        """
        dist_min = float('inf')
        collision = False

        for i, segment in enumerate(self.obstacle_segments):
            xy_start, xy_end = np.array(segment[:2]), np.array(segment[2:])
            closet_dist = point_to_segment_dist(xy_start, xy_end, position) - self.robot.radius

            if closet_dist < 0:
                collision = True
                return collision, dist_min
            elif closet_dist < dist_min:
                dist_min = closet_dist
        
        return collision, dist_min


    def generate_obstacles(self, random_position=False, obstacle_num=3, obstacle_shape='Rectangle'):
        obstacles = []
        if random_position == False:
            # set obstacles with given position, shape, and angles
            pos = [(0, -1), (2, 1), (-1, 2)]
            wxs = [2, 0.1, 0.1]
            wys = [0.1, 2, 1.5]
            angles = [0, 0, np.pi * 3/4]
        else:
            # TODO: put randomly generated obstacles to the environment
            pass

        for i in range(obstacle_num):
            obstacle = Obstacle((pos[i][0], pos[i][1]), wxs[i], wys[i], angles[i])
            obstacles.append(obstacle)

        return obstacles


    def reset(self):
        """
        Set obstacles in env.
        Set robot at (0, -goal_height) with zero initial velocity.
        Return: FullState or ObservableState
        """
        self.global_time = 0
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if self.random_init:
            # set robot position randomly, if collide, reinitialize; randomly initialize orientation too
            # TODO: consider initialize inside an obstacle
            initial_collision = True
            while initial_collision:
                initial_position = np.random.uniform(-self.square_width, self.square_width, size=2)
                initial_collision, _ = self.check_collision(initial_position)
            assert initial_collision is False
            initial_orientation = np.random.uniform(0.0, 2 * np.pi)
        else:
            initial_position = np.array([0, -self.goal_height])
            initial_orientation = 0.0
        
        self.robot.set_state(initial_position[0], initial_position[1], 0, self.goal_height, 0, 0, self.obstacle_segments)
        self.states = list()
        ob = self.robot.observe()

        return ob
    

    def step(self, action, update=True):
        """Compute action for robot, detect collision, update environment.
        action: ActionRotation
        Return: (ob, reward, done, info)
        """
        # collision detection
        if isinstance(action, ActionRotation):
            next_orientation = self.robot.compute_next_orientation(action, self.time_step)
            next_position = self.robot.compute_next_position(next_orientation, action, self.time_step)
        else:
            next_orientation = self.robot.orientation
            next_position = self.robot.compute_next_xy(action, self.time_step)
        collision, dist_min = self.check_collision(next_position)

        # check if reaching the goal
        next_goal_dist = np.linalg.norm(next_position - self.robot.get_goal_position())
        goal_reached = next_goal_dist < self.goal_reaching_radius

        # reward the robot if its achieving to the goal
        #robot_cur_goal_dist = self.robot.get_goal_distance()
        #reward = self.goal_dist_reward_factor if (next_goal_dist - robot_cur_goal_dist) < 0.0 else 0.0
        reward = self.goal_dist_penalty_factor * next_goal_dist
        
        # penalize angular movements for a smooth trajectory
        reward += self.rotation_penalty_factor * np.abs(next_orientation - self.robot.orientation)

        if self.global_time > self.time_limit:
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
            reward += (self.discomfort_dist - dist_min) * self.discomfort_penalty_factor
            done = False
            info = "Danger"
        else:
            done = False
            info = "Nothing"
        
        if update:
            # update agent states
            if isinstance(action, ActionRotation):
                ob = self.robot.step(action, self.obstacle_segments, next_orientation, next_position)
            else:
                ob = self.robot.step_xy(action, self.obstacle_segments, next_position)
            # get full state for plotting
            self.states.append(self.robot.get_full_state())
            self.global_time += self.time_step
        
        return ob, reward, done, info


    def render(self, mode='video', output_file=None):
        #plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'silver'
        laser_color = 'lightskyblue'
        goal_color = 'tomato'
        arrow_color = 'red'
        obstacle_color = 'darkgrey'
        arrow_style = patches.ArrowStyle("simple", head_length=5, head_width=3)

        if mode == 'trajectory':
            pass
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7), facecolor='white', dpi=250)
            ax.tick_params(labelsize=16)
            ax.set_xlim(-self.square_width, self.square_width)
            ax.set_ylim(-self.square_width, self.square_width)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # set robot positions and its directions_vector as a representation of its velocity directions_vector at each time step
            robot_positions = [state.position for state in self.states]
            radius = self.robot.radius
            directions_vector = []

            for state in self.states:
                theta = state.orientation
                offset_x = radius * np.cos(theta)
                offset_y = radius * np.sin(theta)
                directions_vector.append(((state.px - offset_x, state.py - offset_y), (state.px + offset_x, state.py + offset_y)))

            # set obstacles
            obstacles_ = [patches.Rectangle(obs.bl_anchor_point(), obs.wx, obs.wy, obs.angle * 180. / np.pi, color=obstacle_color) for obs in self.obstacles]
            for obstacle_ in obstacles_:
                ax.add_artist(obstacle_)
            
            # get ranger reflections and robot orientation at each time step
            ranger_reflectionss = [state.ranger_reflections for state in self.states]
            angless = []
            for state in self.states:
                angles = np.linspace(state.orientation, state.orientation + self.robot.fov, num=self.robot.num_rangers, endpoint=False)
                angless.append(angles)

            # plot ranger reflections at time step 0
            lasers_ = []
            def plot_ranger_reflections(angless, ranger_reflectionss, frame):
                for theta, reflection in zip(angless[frame], ranger_reflectionss[frame]):
                    laser = matlines.Line2D(xdata=[robot_positions[frame][0] + radius*np.cos(theta), robot_positions[frame][0] + reflection*np.cos(theta)],
                                            ydata=[robot_positions[frame][1] + radius*np.sin(theta), robot_positions[frame][1] + reflection*np.sin(theta)],
                                            color=laser_color, linestyle='-', linewidth=2)
                    lasers_.append(laser)
                for laser in lasers_:
                    ax.add_artist(laser)

            plot_ranger_reflections(angless=angless, ranger_reflectionss=ranger_reflectionss, frame=0)

            # set robot, goal pos, arrows of each robot, ranger reflections and time annotation at timestep 0
            goal = matlines.Line2D(xdata=[0], ydata=[self.goal_height],
                                   color=goal_color, marker="*", linestyle='None', markersize=20, label='Goal')

            robot_ = plt.Circle(robot_positions[0], radius, fill=True, color=robot_color)
            arrows = [patches.FancyArrowPatch(*directions_vector[0], color=arrow_color, arrowstyle=arrow_style)]

            time_annotation = plt.text(-0.5, self.square_width + 0.2, 'Time: {}'.format(0), fontsize=16)

            ax.add_artist(robot_)
            ax.add_artist(goal)
            for arrow in arrows: # only one robot in this case, can be further incorporate with multi-agents
                ax.add_artist(arrow)
            ax.add_artist(time_annotation)
            plt.legend([robot_, goal], ['Robot', 'Goal'], fontsize=16, loc='lower right', fancybox=True, framealpha=0.5)


            def update(frame_num):
                nonlocal arrows
                nonlocal lasers_
                
                # replot ranger reflections
                for laser in lasers_:
                    laser.remove()
                lasers_ = []
                plot_ranger_reflections(angless=angless, ranger_reflectionss=ranger_reflectionss, frame=frame_num)
                
                # update robot position
                robot_.center = robot_positions[frame_num]

                # update robot velocity direction
                for arrow in arrows: # only a robot in this
                    arrow.remove()
                arrows = [patches.FancyArrowPatch(*directions_vector[frame_num], color=arrow_color, arrowstyle=arrow_style)]
                for arrow in arrows:
                    ax.add_artist(arrow)
                
                # add time annotation
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