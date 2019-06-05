import numpy as np


class RandomTrajectoryAgent(object):

    def __init__(self,
                 obs_index_dict,
                 lin_vel_rayleigh_scale=.13,
                 rot_vel_std=330,
                 dt=0.1, #0.02
                 avoid_walls=True,
                 avoidance_weight=.005,
                 n_sensors=36,
                 ):

        self.obs_index_dict = obs_index_dict

        # Initialize simulated head direction
        self.th = 0

        self.lin_vel_rayleigh_scale = lin_vel_rayleigh_scale
        self.rot_vel_std = rot_vel_std
        self.dt = dt

        # If true, modify the random motion to be biased towards avoiding walls
        self.avoid_walls = avoid_walls

        # Note that the two axes are 90 degrees apart to work
        self.avoidance_matrix = np.zeros((2, n_sensors))
        self.avoidance_matrix[0, :] = np.roll(np.linspace(-1., 1., n_sensors), 3*n_sensors//4) * avoidance_weight
        self.avoidance_matrix[1, :] = np.linspace(-1., 1., n_sensors) * avoidance_weight

    def act(self, obs):

        distances = obs[self.obs_index_dict['dist_sensors']]

        lin_vel = np.random.rayleigh(scale=self.lin_vel_rayleigh_scale)
        ang_vel = np.random.normal(loc=0, scale=self.rot_vel_std) * np.pi / 180

        cartesian_vel_x = np.cos(self.th) * lin_vel
        cartesian_vel_y = np.sin(self.th) * lin_vel

        # for testing obstacle avoidance only
        # cartesian_vel_x = 0
        # cartesian_vel_y = 0

        self.th = self.th + ang_vel * self.dt
        if self.th > np.pi:
            self.th -= 2*np.pi
        elif self.th < -np.pi:
            self.th += 2*np.pi

        # optional wall avoidance bias
        if self.avoid_walls:
            vel_change = np.dot(self.avoidance_matrix, distances)
            cartesian_vel_x += vel_change[0]
            cartesian_vel_y += vel_change[1]

        # TODO: add a goal achieving bias? To get a trajectory that covers more area

        return np.array([cartesian_vel_x, cartesian_vel_y]) * 3
