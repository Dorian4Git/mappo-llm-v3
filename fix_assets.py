from PIL import Image
import os
import random

asset_dir = r"c:\PROJECTS\_SCHOOL\MasterIS\TM\mappo-llm-v3\assets"

def make_transparent(filename):
    path = os.path.join(asset_dir, filename)
    if not os.path.exists(path): return
    
    img = Image.open(path).convert("RGBA")
    data = img.getdata()
    
    new_data = []
    # Magenta threshold: R > 200, G < 50, B > 200
    for item in data:
        if item[0] > 150 and item[1] < 100 and item[2] > 150:
            # Check if it's more magenta than white/grey
            if item[0] - item[1] > 80 and item[2] - item[1] > 80:
                new_data.append((255, 255, 255, 0)) # transparent
                continue
        new_data.append(item)
            
    img.putdata(new_data)
    img.save(path, "PNG")
    print(f"Made {filename} transparent.")

sprites = ["agent_0.png", "agent_1.png", "enemy_alive.png", "enemy_dead.png", 
           "wood_resource.png", "stone_resource.png", "iron_resource.png", 
           "gold_resource.png", "workbench.png", "obstacle_rock.png", "bridge_tile.png"]
           
for s in sprites:
    make_transparent(s)

# Generate seamless grass
print("Generating grass_tile.png...")
grass = Image.new("RGBA", (64, 64), (80, 140, 60, 255)) # Base green
pixels = grass.load()
for x in range(64):
    for y in range(64):
        r = random.random()
        if r < 0.1:
            pixels[x,y] = (70, 120, 50, 255) # Darker grass
        elif r < 0.15:
            pixels[x,y] = (90, 160, 70, 255) # Lighter grass
        elif r < 0.16:
            pixels[x,y] = (100, 150, 60, 255) # Little speck

grass.save(os.path.join(asset_dir, "grass_tile.png"))

# Generate seamless water
print("Generating water_tile.png...")
water = Image.new("RGBA", (64, 64), (40, 90, 160, 255)) # Base blue
pixels = water.load()
for x in range(64):
    for y in range(64):
        r = random.random()
        if r < 0.1:
            pixels[x,y] = (50, 100, 180, 255) # Lighter water
        elif r < 0.15:
            pixels[x,y] = (30, 80, 140, 255) # Darker water
        elif r < 0.18:
            pixels[x,y] = (180, 220, 255, 150) # Wave speck
            
water.save(os.path.join(asset_dir, "water_tile.png"))

print("Assets fixed!")
