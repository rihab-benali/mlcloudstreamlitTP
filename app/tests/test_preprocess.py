from PIL import Image
from app.preprocess import preprocess_image

def test_preprocess_output_shape():
    img = Image.new("RGB", (500, 500))
    arr = preprocess_image(img)
    assert arr.shape == (224, 224, 3)
    assert arr.max() <= 1.0
