from model_loader import load_my_model
from PIL import Image
import numpy as np
from preprocess import preprocess_image

if __name__ == "__main__":
    print("🔁 Chargement du modèle...")
    model = load_my_model()

    img = Image.new("RGB", (256, 256))  # Replace this with a real image for better results
    arr = preprocess_image(img)

    input_batch = np.expand_dims(arr, axis=0)  # Shape should be (1, 256, 256, 3)
    print("📏 Input batch shape:", input_batch.shape)

    predictions = model.predict(input_batch)
    print("✅ Prédiction:", predictions)
