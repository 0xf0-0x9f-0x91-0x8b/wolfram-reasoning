import os
from PIL import Image

for f in os.listdir('images'):
    if not f.startswith('K12-1729'):
        continue
    img = Image.open('images/' + f).convert("RGB")
    print(f, img)
