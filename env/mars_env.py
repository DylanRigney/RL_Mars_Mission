import pybullet as p
import pybullet_data
from vehicles.rover import Rover
from utils.terrain import create_mars_terrain

class MarsEnv:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.terrain_id = create_mars_terrain()
        self.rover = Rover()

    def reset(self):
        p.resetSimulation()
        self.terrain_id = create_mars_terrain()
        self.rover.reset()

    def step(self, action):
        self.rover.apply_action(action)
        p.stepSimulation()
        obs = self.rover .get_observation()
        reward = self.compute_reward(obs)
        done = False # TODO: Check if episode is done
        info = {}
        return obs, reward, done, info  
    
    def render(self):
        pass # PyBullet handles rendering

    def sample_action(self):
        return self.rover.sample_action()
    
    def compute_reward(self, obs):
        return 0 # TODO: Implement reward function