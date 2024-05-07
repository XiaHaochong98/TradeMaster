import os
from glob import glob
from PIL import Image

paths = glob(os.path.join("*.png"))
for path in paths:
    img = Image.open(path)
    img = img.resize((250, 250))
    img.save(path)