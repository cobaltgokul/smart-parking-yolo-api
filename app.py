from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import requests
 
app = Flask(__name__)
 
model = YOLO("best.pt")
 
 
@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.json
        area_id = data.get("area_id")
 
        # --- Accept image in two ways ---
        # 1) base64-encoded string  (preferred, sent directly from edge function)
        # 2) a publicly accessible URL (fallback — Supabase signed URLs work here)
 
        image_base64 = data.get("image_base64")
        image_url    = data.get("image_url")
 
        if image_base64:
            # Decode base64 → numpy array
            img_bytes = base64.b64decode(image_base64)
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif image_url:
            # Download image from URL → numpy array
            resp = requests.get(image_url, timeout=15)
            if resp.status_code != 200:
                return jsonify({"error": f"Could not fetch image from URL: {resp.status_code}"}), 400
            np_arr = np.frombuffer(resp.content, np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            return jsonify({"error": "Provide either 'image_base64' or 'image_url' in the request body"}), 400
 
        if frame is None:
            return jsonify({"error": "Failed to decode image. Make sure it is a valid JPG/PNG."}), 400
 
        # Run YOLOv8 inference
        results = model(frame)
 
        slots = []
        slot_number = 1
 
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls        = int(box.cls[0])
                confidence = float(box.conf[0])
 
                # cls 0 = empty, cls 1 = occupied  (adjust if your labels differ)
                status = "occupied" if cls == 1 else "empty"
 
                slots.append({
                    "slot_number": slot_number,
                    "status":      status,
                    "confidence":  round(confidence, 3),
                    "x":           int(x1),
                    "y":           int(y1),
                    "width":       int(x2 - x1),
                    "height":      int(y2 - y1),
                })
                slot_number += 1
 
        empty_count    = sum(1 for s in slots if s["status"] == "empty")
        occupied_count = sum(1 for s in slots if s["status"] == "occupied")
 
        return jsonify({
            "area_id":        area_id,
            "slots":          slots,
            "total_empty":    empty_count,
            "total_occupied": occupied_count,
        })
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "best.pt loaded"})
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)