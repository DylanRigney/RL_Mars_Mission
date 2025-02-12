import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image
from vehicles.rover import Rover
import random

class MarsEnv:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -3.72) # mars gravity
        self.terrain_id = self.create_mars_terrain()
        self.rover = Rover()
        self.goal_position = [random.randint(0, 10), random.randint(0, 10), 10] # goal position
        self.goal_id = p.loadURDF("sphere2.urdf", self.goal_position, globalScaling=0.5)

    def create_mars_terrain(self):
        
        heightmap_path = "./assets/mars_heightmap.png"
        # Open the heightmap image
        heightmap = Image.open(heightmap_path).convert("L")
        heightmap_data = np.array(heightmap)
    
        # Normalize the heightmap data
        heightmap_data = (heightmap_data - heightmap_data.min()) / (heightmap_data.max() - heightmap_data.min())

        # Create the terrain
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[1, 1, 8],
            heightfieldData=heightmap_data.flatten(),
            numHeightfieldRows=heightmap_data.shape[0],
            numHeightfieldColumns=heightmap_data.shape[1]
        )

        terrain_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape)
    
        p.changeVisualShape(terrain_id, -1, rgbaColor=[0.8, 0.4, 0.2, 1])  # Reddish-brown Mars color

        return terrain_id

    def reset(self):
        p.resetSimulation()
        self.terrain_id = self.create_mars_terrain()
        self.rover.reset()

    def step(self, action):
        self.rover.apply_action(action)
        p.stepSimulation()
        obs = self.rover.get_observation()
        reward, done = self.calculate_reward(obs)
        return obs, reward, done

    def sample_action(self):
        return self.rover.sample_action()
    
    def calculate_reward(self, obs):
        rover_pos, _ = p.getBasePositionAndOrientation(self.rover.rover_id)
        distance_to_goal = np.linalg.norm(np.array(rover_pos[:2]) - np.array(self.goal_position[:2]))
        
        # rewaard design
        reward = -distance_to_goal # Closer = higher reward (less negative)
        done = distance_to_goal < .5 # Task is done when the rover reaches the goal

        if done:
            reward += 100 # Add a bonus reward for successfully reaching the goal
        
        return reward, done