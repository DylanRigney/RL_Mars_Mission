import requests

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