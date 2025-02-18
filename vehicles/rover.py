import pybullet as p
import numpy as np
import random

class Rover:
    def __init__(self, x, y, z):
         # Load the Husky rover URDF.
        self.rover_id = p.loadURDF("husky/husky.urdf", basePosition=[x, y, z])
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

        # Normalize position 
        normalized_position = position[0] / 20
        normalized_y = position[1] / 20
        normalized_z = (position[2] / 5) * 2 - 1

        # Normalize orientation
        normalized_roll = orientation_euler[0] / np.pi
        normalized_pitch = orientation_euler[1] / np.pi
        normalized_yaw = orientation_euler[2] / np.pi

        # Normalize velocities
        normalized_vx = linear_velocity[0] / 10
        normalized_vy = linear_velocity[1] / 10
        normalized_vz = linear_velocity[2] / 10

        normalized_wx = angular_velocity[0] / 5
        normalized_wy = angular_velocity[1] / 5
        normalized_wz = angular_velocity[2] / 5

        # Combine into a Single Observation Array
        observation = np.array([
            normalized_position, normalized_y, normalized_z,
            normalized_roll, normalized_pitch, normalized_yaw,
            normalized_vx, normalized_vy, normalized_vz,
            normalized_wx, normalized_wy, normalized_wz
        ])

        return observation

    def sample_action(self):
        # Sample randomly from the discrete action space.
        return np.random.choice(self.action_space)