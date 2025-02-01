class Rover:
    def __init__(self):
        self.id = p.loadURDF("husky/husky.urdf", basePosition=[0,0,2.5]) # Load Husky URDF
        # self.id = p.loadURDF("r2d2.urdf")  # Load R2D2 URDF

    def reset(self):
        p.resetBasePositionAndOrientation(self.id, [0, 0, 2.5], [0, 0, 0, 1])

    def apply_action(self, action):
        raise NotImplementedError # TODO: Implement apply_action
    
    def get_observation(self, rover_id):
        # Get Position and Orientation
        position, orientation = p.getBasePositionAndOrientation(rover_id)
        orientation_euler = p.getEulerFromQuaternion(orientation)

        # Get Linear and Angular Velocity
        linear_velocity, angular_velocity = p.getBaseVelocity(rover_id)

        # Combine into a Single Observation Array
        observation = np.array([
            *position,                  # x, y, z
            *orientation_euler,         # roll, pitch, yaw
            *linear_velocity,           # vx, vy, vz
            *angular_velocity           # wx, wy, wz
        ])

        return observation

    def sample_action(self):
        raise NotImplementedError # TODO: Implement sample_action