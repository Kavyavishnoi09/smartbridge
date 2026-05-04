import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from predict import load_rice_model, predict_rice_type

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

# Default class names. Change these if your dataset folder names are different.
CLASS_NAMES = ["arborio", "basmati", "ipsala", "jasmine", "karacadag"]

MODEL_PATH = os.path.join("training", "rice.h5")
model = load_rice_model(MODEL_PATH)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/details")
def details():
    return render_template("details.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template(
            "results.html",
            error="No image file was uploaded. Please choose an image and try again."
        )

    file = request.files["file"]

    if file.filename == "":
        return render_template(
            "results.html",
            error="No file selected. Please choose a rice grain image."
        )

    if not allowed_file(file.filename):
        return render_template(
            "results.html",
            error="Unsupported file type. Please upload a PNG, JPG, or JPEG image."
        )

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        predicted_class, confidence, all_scores = predict_rice_type(model, filepath, CLASS_NAMES)
        return render_template(
            "results.html",
            prediction=predicted_class,
            confidence=confidence,
            image_file=filename,
            scores=all_scores
        )
    except FileNotFoundError:
        return render_template(
            "results.html",
            error=(
                "Model file not found. Train the model first and place the saved file at "
                "'training/rice.h5'."
            )
        )
    except Exception as exc:
        return render_template(
            "results.html",
            error=f"Prediction failed: {exc}"
        )


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
