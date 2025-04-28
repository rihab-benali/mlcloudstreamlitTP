import numpy as np
from PIL import Image

def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    return arr
