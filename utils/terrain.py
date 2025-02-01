import requests
import pybullet as p
import numpy as np
from PIL import Image

# Download the heightmap image

# Define the URL for the heightmap tile
tile_url = "http://s3-eu-west-1.amazonaws.com/whereonmars.cartodb.net/mola-gray/4/2/2.png"

# Define the local file path to save the image
heightmap_path = "mars_heightmap.png"

# Download the image
response = requests.get(tile_url)

# Save the image if the request was successful
if response.status_code == 200:
    with open(heightmap_path, "wb") as file:
        file.write(response.content)

    print(f"Heightmap downloaded and saved as {heightmap_path}")
else:
    print(f"Failed to download heightmap. Status code: {response.status_code}")


def create_mars_terrain():
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
        numHeightfieldColumns=heightmap_data.shape[1],
        heightfieldType=p.GEOM_HF_BILINEAR # Use bilinear interpolation for smooth terrain (TODO: Check effects)
    )

    terrain_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape)
    p.changeVisualShape(terrain_id, -1, rgbaColor=[0.8, 0.4, 0.2, 1])  # Reddish-brown Mars color

    return terrain_id