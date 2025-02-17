import pybullet as p
import numpy as np
import random

class Rover:
    def __init__(self):
         # Load the Husky rover URDF.
        self.rover_id = p.loadURDF("husky/husky.urdf", basePosition=[random.randint(0, 10), random.randint(0, 10), 2.5])
        # Wheel joint indices for Husky.
        self.wheels = [2, 3, 4, 5]
        # Define a discrete action space: 0: Forward, 1: Turn Left, 2: Turn Right, 3: Stop.
        self.action_space = [0, 1, 2, 3]


    def reset(self, position):
        p.resetBasePositionAndOrientation(self.rover_id, position, [0, 0, 0, 1])

    def apply_action(self, action):
        max_force = 100
        target_velocity = 10 

        # Discrete Action Space:
        if action == 0:  # Move Forward
            wheel_velocities = [target_velocity] * 4
        elif action == 1:  # Turn Left
            wheel_velocities = [target_velocity, target_velocity, -target_velocity, -target_velocity]
        elif action == 2:  # Turn Right
            wheel_velocities = [-target_velocity, -target_velocity, target_velocity, target_velocity]
        elif action == 3:  # Stop
            wheel_velocities = [0] * 4
        else:
            raise ValueError("Invalid Action")
        
        for wheel, velocity in zip(self.wheels, wheel_velocities):
            p.setJointMotorControl2(bodyUniqueId=self.rover_id,
                                    jointIndex=wheel,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=velocity,
                                    force=max_force)
    
    def get_observation(self):
        # Get Position and Orientation
        position, orientation = p.getBasePositionAndOrientation(self.rover_id)
        orientation_euler = p.getEulerFromQuaternion(orientation)

        # Get Linear and Angular Velocity
        linear_velocity, angular_velocity = p.getBaseVelocity(self.rover_id)

        # Combine into a Single Observation Array
        observation = np.array([
            *position,                  # x, y, z
            *orientation_euler,         # roll, pitch, yaw
            *linear_velocity,           # vx, vy, vz
            *angular_velocity           # wx, wy, wz
        ])

        return observation

    def sample_action(self):
        # Sample randomly from the discrete action space.
        return np.random.choice(self.action_space)