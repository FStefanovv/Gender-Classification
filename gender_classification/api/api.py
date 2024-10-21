from flask import Flask, request, jsonify
import numpy as np
import cv2
from classifier import GenderClassifier
from image_processing import get_faces
from utils import tag_faces

app = Flask(__name__)

classifier = GenderClassifier()


@app.route("/classify-people-gender", methods=["POST"])
def classify_faces():
    if "image_file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    image_file = request.files["image_file"]

    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image_data = image_file.read()
        nparr = np.frombuffer(image_data, np.uint8)

        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        faces, rects = get_faces(image)

        classified_faces = classifier.classify(faces)

        tag_faces(image, classified_faces, rects)

        _, buffer = cv2.imencode(".jpg", image)
        return buffer.tobytes(), 200, {"Content-Type": "image/jpeg"}

    except ValueError as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
