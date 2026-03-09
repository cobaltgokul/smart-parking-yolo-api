from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")


@app.route("/detect", methods=["POST"])
def detect():

    data = request.json

    video_url = data.get("video_url")
    area_id = data.get("area_id")

    cap = cv2.VideoCapture(video_url)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Unable to read video"}), 400

    results = model(frame)

    slots = []
    slot_number = 1

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])

            status = "empty"
            if cls == 1:
                status = "occupied"

            slots.append({
                "slot_number": slot_number,
                "status": status,
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            })

            slot_number += 1

    return jsonify({
        "area_id": area_id,
        "slots": slots
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)