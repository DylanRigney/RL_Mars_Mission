import pybullet as p
import time
import pybullet_data
import numpy as np
from PIL import Image

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


# Setup heightmap
# -----------------------------------------------------------------------------------------------

# Set the path to your heightmap image
heightmap_path = "assets/mars_heightmap.png"

# Load the heightmap image using PIL
heightmap_image = Image.open(heightmap_path)

# Convert the image to grayscale (if it's not already)
heightmap_image = heightmap_image.convert("L")

# Get the dimensions of the image
width, height = heightmap_image.size

# Get the pixel values as a NumPy array
heightmap_data = np.asarray(heightmap_image, dtype=np.float32)

# Normalize the heightmap data to the range [0, 1]
heightmap_data /= 255.0

# Create a heightfield shape
terrain_shape = p.createCollisionShape(
    shapeType=p.GEOM_HEIGHTFIELD,
    meshScale=[0.1, 0.1, 0.1],
    heightfieldTextureScaling=(width - 1) / 2,
    heightfieldData=heightmap_data.flatten(),
    numHeightfieldRows=height,
    numHeightfieldColumns=width
)    


# Setup environment
# -----------------------------------------------------------------------------------------------

# Create a multi-body object for the terrain
terrain = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape, basePosition=[0, 0, 0])

p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

# mars gravity
p.setGravity(0, 0, -3.72)

# Rover Setup
# -----------------------------------------------------------------------------------------------

roverId = p.loadURDF("husky/husky.urdf", basePosition=[0,0,0.2])

# Key mapping
keys = {p.B3G_UP_ARROW: "forward", p.B3G_DOWN_ARROW: "backward",
        p.B3G_LEFT_ARROW: "left", p.B3G_RIGHT_ARROW: "right"}

print("Start here---------------------------------------------------------------------------------")

""" Get joint information
 for joint_index in range(p.getNumJoints(roverId)):
    joint_info = p.getJointInfo(roverId, joint_index)
    print(f"Joint {joint_index}: {joint_info[1].decode('utf-8')}") """

wheel_joints = [2,3,4,5]
max_force = 10  # Adjust for stronger or weaker motor response

for joint in wheel_joints:
    p.setJointMotorControl2(roverId, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

while True:
    p.stepSimulation()
    
    # Read keyboard events
    keys_pressed = p.getKeyboardEvents()
    
    # Set default velocity to 0 (stop movement)
    target_velocity = 0
    turn_factor = 0
    
    # Adjust speed based on key presses
    if p.B3G_UP_ARROW in keys_pressed and keys_pressed[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        target_velocity = 5  # Move forward
    elif p.B3G_DOWN_ARROW in keys_pressed and keys_pressed[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        target_velocity = -5  # Move backward
    
    # Adjust turning
    if p.B3G_LEFT_ARROW in keys_pressed and keys_pressed[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        turn_factor = -2  # Turn left
    elif p.B3G_RIGHT_ARROW in keys_pressed and keys_pressed[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        turn_factor = 2  # Turn right

    # Apply motor controls
    p.setJointMotorControl2(roverId, wheel_joints[0], p.VELOCITY_CONTROL, targetVelocity=target_velocity + turn_factor, force=max_force)
    p.setJointMotorControl2(roverId, wheel_joints[1], p.VELOCITY_CONTROL, targetVelocity=target_velocity - turn_factor, force=max_force)
    p.setJointMotorControl2(roverId, wheel_joints[2], p.VELOCITY_CONTROL, targetVelocity=target_velocity + turn_factor, force=max_force)
    p.setJointMotorControl2(roverId, wheel_joints[3], p.VELOCITY_CONTROL, targetVelocity=target_velocity - turn_factor, force=max_force)

    time.sleep(1./240.)  # Maintain real-time simulation speed
