import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image
from vehicles.rover import Rover
import random
from gym import spaces

class MarsEnv:
    def __init__(self):
        # Connect to PyBullet (if not already connected)
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -3.72) # mars gravity

        # Create the terrain using a heightmap
        self.terrain_id = self.create_mars_terrain()

        # Create the rover (this loads the Husky model)
        self.rover = Rover()

        # Define a random goal position (adjust as needed)
        self.goal_position = [random.randint(0, 10), random.randint(0, 10), 10] # goal position
        self.goal_id = p.loadURDF("sphere2.urdf", self.goal_position, globalScaling=0.5)

        # Define the observation space and action space
        obs = self.rover.get_observation()
        self.obs_dim = len(obs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Action
        self.action_space = spaces.Discrete(len(self.rover.action_space))

    def create_mars_terrain(self):
        
        heightmap_path = "./assets/mars_heightmap.png"
        # Open the heightmap image
        heightmap = Image.open(heightmap_path).convert("L")
        heightmap_data = np.array(heightmap)
    
        # Normalize the heightmap data
        heightmap_data = (heightmap_data - heightmap_data.min()) / (heightmap_data.max() - heightmap_data.min())

        # Create the terrain collision shape
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[1, 1, 8],
            heightfieldData=heightmap_data.flatten(),
            numHeightfieldRows=heightmap_data.shape[0],
            numHeightfieldColumns=heightmap_data.shape[1]
        )

        terrain_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape)
        p.changeVisualShape(terrain_id, -1, rgbaColor=[0.8, 0.4, 0.2, 1])  # Mars-like color

        return terrain_id

    def reset(self):
        # Reset the simulation and reinitialize environment objects
        p.resetSimulation()
        self.terrain_id = self.create_mars_terrain()

        # Recreate the rover and goal after reset:
        self.rover.reset()
        self.goal_position = [random.randint(0, 10), random.randint(0, 10), 10]
        self.goal_id = p.loadURDF("sphere2.urdf", self.goal_position, globalScaling=0.5)
        
        # Return initial observation
        return self.rover.get_observation()

    def step(self, action):
        # Apply the given action on the rover
        self.rover.apply_action(action)
        p.stepSimulation()
        obs = self.rover.get_observation()
        reward, done = self.calculate_reward(obs)
        
        return obs, reward, done, {} # Extra info dict for compatibility

    def sample_action(self):
        return self.rover.sample_action()
    
    def calculate_reward(self, obs):
        rover_pos, _ = p.getBasePositionAndOrientation(self.rover.rover_id)
         # Compute Euclidean distance (only considering x and y coordinates)
        distance_to_goal = np.linalg.norm(np.array(rover_pos[:2]) - np.array(self.goal_position[:2]))
        
        # Reward: Negative distance penalty; bonus upon reaching goal
        reward = -distance_to_goal # Closer = higher reward (less negative)
        done = distance_to_goal < .5 # Task is done when the rover reaches the goal

        if done:
            reward += 100 # Bonus reward for reaching the goal
        
        return reward, done

    def render(self):
        # Optionally, implement a render method to visualize or save screenshots.
        pass

    def close(self):
        p.disconnect()