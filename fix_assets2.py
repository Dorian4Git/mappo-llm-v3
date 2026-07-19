from PIL import Image
import os

asset_dir = r"c:\PROJECTS\_SCHOOL\MasterIS\TM\mappo-llm-v3\assets"

def clean_magenta(filename):
    path = os.path.join(asset_dir, filename)
    if not os.path.exists(path): return
    
    img = Image.open(path).convert("RGBA")
    data = img.getdata()
    
    new_data = []
    for item in data:
        r, g, b, a = item
        # If it has more red and blue than green, it's a shade of magenta
        if r > 80 and b > 80 and g < r - 20 and g < b - 20:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
            
    img.putdata(new_data)
    img.save(path, "PNG")
    print(f"Cleaned {filename}")

clean_magenta("workbench.png")
clean_magenta("gold_resource.png")
