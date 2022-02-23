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
        self.time_step = 0.1 # in seconds
        self.global_time = 0 # in seconds
        self.random_init = True # randomly initialize robot position
        self.robot_po = True # partial observability of the robot
        self.random_obstacle = True
        self.obstacle_num = None
        self.training_stage = 'first'
        self.eval = False
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
        #self.goal_dist_reward_factor = 1.0
        #self.rotation_penalty_factor = -0.0
        self.step_penalty = -0.1
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = -5.0

        # goal reaching radius: 0.5m
        self.goal_reaching_radius = 10 * self.robot.radius

        # simulation config
        self.square_width = 4.0 # half width of the square environment
        self.goal_height = 3.0 # initial distance to goal pos
        self.UP_RIGHT = (self.square_width, self.square_width)
        self.BOTTOM_RIGHT = (self.square_width, -self.square_width)
        self.BOTTOM_LEFT = (-self.square_width, -self.square_width)
        self.UP_LEFT = (-self.square_width, self.square_width)

        self.front_wall = (self.UP_LEFT[0], self.UP_LEFT[1], self.UP_RIGHT[0], self.UP_RIGHT[1])
        self.right_wall = (self.UP_RIGHT[0], self.UP_RIGHT[1], self.BOTTOM_RIGHT[0], self.BOTTOM_RIGHT[1])
        self.back_wall = (self.BOTTOM_RIGHT[0], self.BOTTOM_RIGHT[1], self.BOTTOM_LEFT[0], self.BOTTOM_LEFT[1])
        self.left_wall = (self.BOTTOM_LEFT[0], self.BOTTOM_LEFT[1], self.UP_LEFT[0], self.UP_LEFT[1])
        self.walls = [self.front_wall, self.right_wall, self.back_wall, self.left_wall] # ((x1, y1), (x1', y1'))

        #self.obstacles = self.generate_obstacles(random_position=self.random_obstacle, obstacle_num=self.obstacle_num)
        #self.obstacle_segments = self.walls + self.obstacles2segments(self.obstacles) # list of segments (x1, y1, x1', y1')

        # visualization, store full state of the robot
        self.states = None


    def _set_robot(self, robot):
        self.robot = robot
        self.robot.time_step = self.time_step


    def set_training_stage(self, training_stage):
        self.training_stage = training_stage


    def enable_eval_mode(self, eval):
        self.eval = bool(eval)


    def enable_random_obstacle(self, enabled):
        self.random_obstacle = bool(enabled)


    def set_obstacle_num(self, num):
        self.obstacle_num = num


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

            if closet_dist <= 0.0:
                collision = True
                return collision, dist_min
            elif closet_dist < dist_min:
                dist_min = closet_dist
        
        return collision, dist_min


    def check_far_enough(self, position):
        dist_min = float('inf')
        too_close = False

        for i, segment in enumerate(self.obstacle_segments):
            xy_start, xy_end = np.array(segment[:2]), np.array(segment[2:])
            # make sure robot is intially set at 1m away from any obstacles
            closet_dist = point_to_segment_dist(xy_start, xy_end, position) - 20 * self.robot.radius

            if closet_dist <= 0.0:
                too_close = True
                return too_close, dist_min
            elif closet_dist < dist_min:
                dist_min = closet_dist
        
        return too_close, dist_min


    def generate_obstacles(self, random_position, obstacle_num, training_stage='first', eval=False):
        obstacles = []
        if not random_position:
            # set obstacles with given position, shape, and angles
            #obstacle_list = [[1.72, -0.06, 0.73, 0.23], [0.05, 0.2, 0.52, 0.51], [0.69, -0.91, 0.47, 0.22], [-1.71, 0.06, 0.81, 0.22], [2.07, 1.13, 0.52, 0.41], [1.36, -0.24, 0.7, 0.47]]
            #obstacle_list = [[-2.63, 0.96, 0.4, 0.36], [-1.93, -0.31, 0.55, 0.5], [-0.12, 0.09, 0.7, 0.24], [2.68, 0.29, 0.81, 0.68], [-1.92, -0.34, 0.79, 0.37], [0.86, 1.0, 0.44, 0.26]]
            #obstacle_list = [[2.76, -0.98, 0.58, 0.44], [-0.0, -1.47, 0.63, 0.4], [1.77, -0.62, 0.4, 0.7], [0.19, 0.78, 0.85, 0.31], [-2.58, -0.56, 0.52, 0.46], [-1.44, -0.8, 0.9, 0.27], [0.83, 1.37, 0.44, 0.55]]
            obstacle_list = [[-1.91, -1.13, 0.5, 0.38], [-2.77, -0.22, 0.95, 0.42], [2.64, -1.17, 0.98, 0.51], [3.31, -0.41, 0.86, 0.38], [0.11, 0.09, 0.6, 0.59], [2.1, -2.78, 0.4, 0.45], [-1.5, -2, 0.8, 0.8], [-1.5, 2, 0.5, 0.8], [2.5, 1, 0.5, 0.38], [1.0, -1, 0.6, 0.7]]
            obs = [*zip(*obstacle_list)]
            x = [ob[0] for ob in obstacle_list]
            y = [ob[1] for ob in obstacle_list]
            poses = [*zip(x, y)]
            wxs = [ob[2] for ob in obstacle_list]
            wys = [ob[3] for ob in obstacle_list]
            angles = [0 for _ in range(len(obstacle_list))]
            random_obstacle_num = len(obstacle_list)
            print(random_obstacle_num)
        else:
            old_obstacle_num = obstacle_num
            if training_stage == 'second':
                obstacle_num = 2 * obstacle_num
            if not eval:
                random_obstacle_num = np.random.randint(max(0, obstacle_num - old_obstacle_num), obstacle_num + 1) # max for when given obstacle_num < 5
            elif eval:
                random_obstacle_num = self.obstacle_num # fixed number of obstacles
            poses = [np.random.uniform(low=(-3.5, -3), high=(3.5, 1.5), size=(2,)) for _ in range(random_obstacle_num)]
            wxs = [np.random.uniform(low=0.4, high=1.0) for _ in range(random_obstacle_num)]
            wys = [np.random.uniform(low=0.2, high=0.8) for _ in range(random_obstacle_num)]
            angles = [0 for i in range(random_obstacle_num)]

        for i in range(random_obstacle_num):
            obstacle = Obstacle((poses[i][0], poses[i][1]), wxs[i], wys[i], angles[i])
            obstacles.append(obstacle)

        return obstacles, random_obstacle_num


    def reset(self):
        """
        Set obstacles in env.
        Set robot at (0, -goal_height) with zero initial velocity.
        training state: 'first' or 'second'; if second, obstacle_num = obstacle_num * 2
        Return: FullState or ObservableState
        """
        self.global_time = 0
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if self.random_init:
            self.obstacles, obs_num = self.generate_obstacles(random_position=self.random_obstacle, obstacle_num=self.obstacle_num, training_stage=self.training_stage, eval=self.eval)
            self.obstacle_segments = self.walls + self.obstacles2segments(self.obstacles) # list of segments (x1, y1, x1', y1')
            too_close = True

            while too_close:
                initial_position = np.random.uniform(low=(-2.0, -3.5), high=(2.0, 3.5), size=(2,))
                too_close, _ = self.check_far_enough(initial_position)
            assert too_close is False
            #initial_orientation = np.random.uniform(0.0, 2 * np.pi)
            initial_orientation = 0.0

            if self.eval:
                initial_position = np.array([0, -self.goal_height])
            goal_position = np.random.uniform(low=(-0.5, 2.0), high=(0.5, 3.0), size=(2,))
        else:
            initial_position = np.array([0, -self.goal_height])
            initial_orientation = 0.0
            goal_position = np.array([0, self.goal_height])
        
        self.robot.set_state(initial_position[0], initial_position[1], goal_position[0], goal_position[1], 0, 0, self.obstacle_segments)
        self.states = list()
        ob = self.robot.observe()

        return ob, obs_num
    

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
            #next_orientation = self.robot.orientation
            next_position = self.robot.compute_next_xy(action, self.time_step)
        collision, dist_min = self.check_collision(next_position)

        # check if reaching the goal
        next_goal_dist = np.linalg.norm(next_position - self.robot.get_goal_position())
        goal_reached = next_goal_dist < self.goal_reaching_radius

        # reward the robot if its achieving to the goal
        #robot_cur_goal_dist = self.robot.get_goal_distance()
        #reward = self.goal_dist_reward_factor * (robot_cur_goal_dist - next_goal_dist)
        #reward = self.goal_dist_penalty_factor * next_goal_dist

        # penalty the robot when time step + 1
        reward = self.step_penalty
        
        # penalize angular movements for a smooth trajectory
        #reward += self.rotation_penalty_factor * np.abs(next_orientation - self.robot.orientation)

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
            info = "Reached"
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


    def render(self, mode='video', output_file=None, density=None):
        #plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'aquamarine'
        laser_color = 'lightskyblue'
        goal_color = 'tomato'
        arrow_color = 'red'
        obstacle_color = 'dodgerblue'
        arrow_style = patches.ArrowStyle("simple", head_length=5, head_width=3)

        # for text plot around robot
        x_offset = 0.11
        y_offset = 0.11

        if mode == 'trajectory':
            fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
            ax.tick_params(labelsize=16)
            ax.set_xlim(-self.square_width, self.square_width)
            ax.set_ylim(-self.square_width, self.square_width)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # set goal point
            goal = matlines.Line2D(xdata=[self.robot.gx], ydata=[self.robot.gy],
                                   color=goal_color, marker="*", linestyle='None', markersize=20, label='Goal')
            ax.add_artist(goal)

            # set obstacles
            obstacles_ = [patches.Rectangle(obs.bl_anchor_point(), obs.wx, obs.wy, obs.angle * 180. / np.pi, color=obstacle_color) for obs in self.obstacles]
            for obstacle_ in obstacles_:
                ax.add_artist(obstacle_)

            robot_positions = [state.position for state in self.states]

            for k in range(len(self.states)):
                # plot robot positions
                if k % density == 0 or k == len(self.states) - 1:
                    robot_ = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    ax.add_artist(robot_)
                
                # plot time in seconds
                global_time = k * self.time_step
                if global_time % 2 == 0 or k == len(self.states) - 1:
                    times = plt.text(robot_.center[0] - x_offset, robot_.center[1] - y_offset, '{:.1f}'.format(global_time))
                    ax.add_artist(times)
                
                # connecting dots
                if k != 0:
                    nav_directions = plt.Line2D((self.states[k - 1].px, self.states[k].px),
                                                (self.states[k - 1].py, self.states[k].py),
                                                color=robot_color, ls='solid')
                    ax.add_artist(nav_directions)
            plt.legend([robot_, goal], ['Robot', 'Goal'], fontsize=16, loc='lower right', fancybox=True, framealpha=0.5)

            if output_file is not None:
                fig.savefig(output_file + '.png')
            else:
                plt.show()
            

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
                #theta = np.arctan2(state.vy, state.vx)
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

            lasers_ = []
            def plot_ranger_reflections(angless, ranger_reflectionss, frame):
                for angle, reflection in zip(angless[frame], ranger_reflectionss[frame]):
                    laser = matlines.Line2D(xdata=[robot_positions[frame][0] + radius*np.cos(angle), robot_positions[frame][0] + reflection*np.cos(angle)],
                                            ydata=[robot_positions[frame][1] + radius*np.sin(angle), robot_positions[frame][1] + reflection*np.sin(angle)],
                                            color=laser_color, linestyle='-', linewidth=2)
                    lasers_.append(laser)
                for laser in lasers_:
                    ax.add_artist(laser)

            # plot ranger reflections at time step 0
            plot_ranger_reflections(angless=angless, ranger_reflectionss=ranger_reflectionss, frame=0)

            # set robot, goal pos, arrows of each robot, ranger reflections and time annotation at timestep 0
            goal = matlines.Line2D(xdata=[self.robot.gx], ydata=[self.robot.gy],
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
                anim.save(output_file + '.gif', writer=writer)
            else:
                plt.show()