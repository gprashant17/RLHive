from gym.spaces import Discrete, Box
from hive.envs import BaseEnv
from hive.envs.env_spec import EnvSpec
import numpy as np

class CrystalEnv(BaseEnv):
    def __init__(self, state_size = 219, n_vocab = 10, n_sites = 14, species_ind = None, 
                 atom_num_dict = None, env_name = 'CrystalEnv', seed = 42, **kwargs):
        
        self.env_name = env_name
        self.state_size = state_size
        self.species_ind = species_ind
        self.atom_num_dict = atom_num_dict
        self.n_vocab = n_vocab

        self.action_space = Discrete(self.n_vocab)
        self.observation_space = Box(low = np.array([-np.inf] * self.state_size), high = np.array([np.inf] * self.state_size))
        self.state = self.random_initial_state()
        self.n_sites = n_sites
        self.t = 0
        self._seed = seed
        self._env_spec = self.create_env_spec(self.env_name, **kwargs)

    def random_initial_state(self):
        state = np.zeros(self.state_size)
        return state

    def calc_energy(self, state, species, atom_num_dict):
        energy = 0
        return energy

    def calc_reward(self, energy):
        reward = 0
        
        ### trial
        reward = np.random.normal(0.0, 1.)
        ###
        
        return reward

    def create_env_spec(self, env_name, **kwargs):
        """
        Each family of environments have their own type of observations and actions.
        You can add support for more families here by modifying observation_space and action_space.
        """
        return EnvSpec(
            env_name=env_name,
            observation_space=[self.observation_space],
            action_space=[self.action_space],
        )
    def reset(self):
        self.t = 0
        self.state = self.random_initial_state()
        return self.state, None

    def step(self, action):

        done = False
        if self.t < self.n_sites - 1:
            pos1 = 9 + (self.n_vocab + 1 + 3) * self.n_sites + self.t
            self.state[pos1] = 0
            self.state[pos1 + 1] = 1
            pos2 = 9 + (self.n_vocab + 1 + 3) * self.t 
            self.state[pos2 : pos2 + self.n_vocab + 1][action] = 1
            self.state[pos2 : pos2 + self.n_vocab + 1][-1] = 0
            reward = 0
            self.t += 1
        else:
            done = True
            reward = self.calc_reward(self.calc_energy(self.state, self.species_ind, self.atom_num_dict))

        info = {}
        return (
            self.state, 
            reward,
            done,
            None,
            info
        )
    def seed(self, seed = 0):
        self._seed = seed
    def close(self):
        pass
