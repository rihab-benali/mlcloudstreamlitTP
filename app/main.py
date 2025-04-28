from model_loader import load_my_model
from PIL import Image
import numpy as np

if __name__ == "__main__":
    print("🔁 Chargement du modèle...")
    model = load_my_model()

    # Dummy test avec une image random (à remplacer par une vraie image si besoin)
    img = Image.new("RGB", (256, 256))
    from preprocess import preprocess_image
    arr = preprocess_image(img)

    input_batch = np.expand_dims(arr, axis=0)
    predictions = model.predict(input_batch)
    print("✅ Prédiction:", predictions)
