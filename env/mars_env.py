import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image
from vehicles.rover import Rover
import random
from math import dist
from gym import spaces

class MarsEnv:
    def __init__(self):
        # Connect to PyBullet (if not already connected)
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -3.72) # mars gravity

        # Create the terrain using a heightmap
        self.terrain_id = self.create_mars_terrain()

        # Create the rover and set its location (this loads the Husky model) 
        rover_x, rover_y, rover_z = random.randint(-6, 6), random.randint(-6, 6), .5
        self.rover = Rover(rover_x, rover_y, rover_z)   

        # Define a random goal position (adjust as needed)
        goal_x, goal_y = random.randint(-6, 6), random.randint(-6, 6)
        self.goal_id = p.loadURDF("sphere2.urdf", [goal_x, goal_y, 2], globalScaling=2)

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
            # Adjust terrain as model improves
            meshScale=[.2, .2, .2],
            heightfieldData=heightmap_data.flatten(),
            numHeightfieldRows=heightmap_data.shape[0],
            numHeightfieldColumns=heightmap_data.shape[1]
        )

        terrain_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape)
        p.changeVisualShape(terrain_id, -1, rgbaColor=[0.8, 0.4, 0.2, 1])  # Mars-like color

        # Create boundaries around the terrain
        self.create_boundaries()

        return terrain_id
    

    def reset(self):
        # Make sure rover and goal don't start too close
        while True:
            rover_x, rover_y = random.randint(-6, 6), random.randint(-6, 6)
            self.goal_x, self.goal_y = random.randint(-6, 6), random.randint(-6, 6)
            
            if dist([rover_x, rover_y], [self.goal_x, self.goal_y]) > 2:
                break

        # reposition the rover and goal positions:
        self.rover.reset([rover_x, rover_y, 1])
        p.resetBasePositionAndOrientation(self.goal_id, [self.goal_x, self.goal_y, 2], [0, 0, 0, 1])

        return self.rover.get_observation()

    def step(self, action, step_count):
        # Apply the given action on the rover
        self.rover.apply_action(action)
        p.stepSimulation() 
        obs = self.rover.get_observation()
        reward, done = self.calculate_reward(obs, step_count)
        
        return obs, reward, done, {} # Extra info dict for compatibility

    def sample_action(self):
        return self.rover.sample_action()
    
    def calculate_reward(self, obs, step_count, max_steps=10000):
        rover_pos, _ = p.getBasePositionAndOrientation(self.rover.rover_id)
        goal_pos, _ = p.getBasePositionAndOrientation(self.goal_id)
        distance_to_goal = np.linalg.norm(np.array(rover_pos[:2]) - np.array(goal_pos[:2]))
        
        # Reward: Negative distance penalty; bonus upon reaching goal
        reward = -distance_to_goal # Closer = higher reward (less negative)
        
        # Check if the rover has reached the goal
        done = distance_to_goal < 3

        # Also terminate the episode if it has taken too many steps
        if step_count >= max_steps:
            done = True
            reward -= 50  # Penalty for not reaching the goal quickly

        if done:
            reward += 100  # Bonus reward for reaching the goal

        return reward, done

    def render(self):
        # Optionally, implement a render method to visualize or save screenshots.
        pass

    def close(self):
        p.disconnect()


    def create_boundaries(self, map_width=15, map_length=15, wall_thickness=0.2, wall_height=1):
        """
        Creates invisible boundary walls around a rectangular map.
    
        Parameters:
        - map_width: The width of the map in the x-direction.
        - map_length: The length of the map in the y-direction.
        - wall_thickness: Half-thickness of the wall (the collision shape uses half-extents).
        - wall_height: The full height of the wall.
        """
        # Compute map boundaries assuming map is centered at (0,0)
        x_min = -map_width / 2.0
        x_max = map_width / 2.0
        y_min = -map_length / 2.0
        y_max = map_length / 2.0

        # Compute positions for each wall
        left_wall_x = x_min - wall_thickness
        right_wall_x = x_max + wall_thickness
        bottom_wall_y = y_min - wall_thickness
        top_wall_y = y_max + wall_thickness

        # Middle points for y and x
        mid_y = (y_min + y_max) / 2.0
        mid_x = (x_min + x_max) / 2.0

        left_wall_pos = [left_wall_x, mid_y, wall_height / 2.0]
        right_wall_pos = [right_wall_x, mid_y, wall_height / 2.0]
        bottom_wall_pos = [mid_x, bottom_wall_y, wall_height / 2.0]
        top_wall_pos = [mid_x, top_wall_y, wall_height / 2.0]

        # Create Left Wall (vertical at x_min - wall_thickness)
        left_wall_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[wall_thickness, map_length / 2.0, wall_height / 2.0]
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=left_wall_collision,
            baseVisualShapeIndex=-1,  # Invisible wall
            basePosition=left_wall_pos
        )

        # Create Right Wall (vertical at x_max + wall_thickness)
        right_wall_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[wall_thickness, map_length / 2.0, wall_height / 2.0]
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=right_wall_collision,
            baseVisualShapeIndex=-1,
            basePosition=right_wall_pos
        )

        # Create Bottom Wall (horizontal at y_min - wall_thickness)
        bottom_wall_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[map_width / 2.0, wall_thickness, wall_height / 2.0]
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=bottom_wall_collision,
            baseVisualShapeIndex=-1,
            basePosition=bottom_wall_pos
        )

        # Create Top Wall (horizontal at y_max + wall_thickness)
        top_wall_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[map_width / 2.0, wall_thickness, wall_height / 2.0]
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=top_wall_collision,
            baseVisualShapeIndex=-1,
            basePosition=top_wall_pos
        )