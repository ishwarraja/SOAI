import os
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# ----------------- Setup Flask -----------------
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------- Load Pre-trained Model -----------------
model = MobileNetV2(weights='imagenet')  # pretrained on ImageNet

# ----------------- Routes -----------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None

    if request.method == "POST":
        if "photo" not in request.files:
            return "No file uploaded", 400

        file = request.files["photo"]
        if file.filename == "":
            return "No selected file", 400

        # Save uploaded file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        image_url = f"/uploads/{file.filename}"

        try:
            # Load and preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Predict
            preds = model.predict(x)
            decoded = decode_predictions(preds, top=3)[0]

            # Filter only cat, dog, elephant classes
            relevant_classes = []
            for _, class_name, prob in decoded:
                if any(animal in class_name.lower() for animal in ["cat", "dog", "elephant"]):
                    relevant_classes.append(f"{class_name} ({prob*100:.2f}%)")

            if relevant_classes:
                result = "Detected: " + ", ".join(relevant_classes)
            else:
                result = f"No cat, dog, or elephant detected. Top prediction: {decoded[0][1]} ({decoded[0][2]*100:.2f}%)"

        except Exception as e:
            result = f"Error analyzing image: {str(e)}"

    return render_template("index.html", result=result, image_url=image_url)

# ----------------- Run server -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
