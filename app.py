from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import requests
import tempfile
import os
from collections import defaultdict
 
app = Flask(__name__)
model = YOLO("best.pt")
 
 
def decode_image_base64(image_base64: str):
    img_bytes = base64.b64decode(image_base64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
 
 
def download_bytes(url: str):
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise ValueError(f"Could not fetch file from URL (HTTP {resp.status_code})")
    return resp.content
 
 
def run_yolo_on_frame(frame):
    results = model(frame)
    slots = []
    slot_number = 1
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls        = int(box.cls[0])
            confidence = float(box.conf[0])
            # cls 0 = empty, cls 1 = occupied — adjust if your labels differ
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
    return slots
 
 
def majority_vote(all_frame_slots: list):
    """Combine results from multiple frames using majority vote per slot."""
    if not all_frame_slots:
        return []
 
    # Use the frame with the most detections as reference slot count
    slot_count = len(max(all_frame_slots, key=len))
 
    status_votes = defaultdict(lambda: {"empty": 0, "occupied": 0})
    coord_accum  = defaultdict(lambda: {"x": [], "y": [], "width": [], "height": [], "confidence": []})
 
    for frame_slots in all_frame_slots:
        for i, slot in enumerate(frame_slots):
            status_votes[i][slot["status"]] += 1
            coord_accum[i]["x"].append(slot["x"])
            coord_accum[i]["y"].append(slot["y"])
            coord_accum[i]["width"].append(slot["width"])
            coord_accum[i]["height"].append(slot["height"])
            coord_accum[i]["confidence"].append(slot["confidence"])
 
    final_slots = []
    for i in range(slot_count):
        votes  = status_votes[i]
        coords = coord_accum[i]
        status = "occupied" if votes["occupied"] >= votes["empty"] else "empty"
        final_slots.append({
            "slot_number": i + 1,
            "status":      status,
            "confidence":  round(sum(coords["confidence"]) / len(coords["confidence"]), 3),
            "x":           int(sum(coords["x"])      / len(coords["x"])),
            "y":           int(sum(coords["y"])      / len(coords["y"])),
            "width":       int(sum(coords["width"])  / len(coords["width"])),
            "height":      int(sum(coords["height"]) / len(coords["height"])),
        })
    return final_slots
 
 
def extract_frames(video_bytes: bytes, num_frames: int = 5):
    """Extract evenly spaced frames from video bytes."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
 
    frames = []
    try:
        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
        if total <= 0:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        else:
            positions = [int(total * i / num_frames) for i in range(num_frames)]
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        cap.release()
    finally:
        os.unlink(tmp_path)
 
    return frames
 
 
@app.route("/detect", methods=["POST"])
def detect():
    try:
        data      = request.json
        area_id   = data.get("area_id")
        file_type = data.get("file_type", "image")   # "image" or "video"
 
        image_base64 = data.get("image_base64")
        image_url    = data.get("image_url")
 
        if file_type == "video":
            # ── VIDEO: download → extract 5 frames → YOLO → majority vote ──
            if not image_url:
                return jsonify({"error": "Provide 'image_url' for video files"}), 400
 
            video_bytes = download_bytes(image_url)
            frames      = extract_frames(video_bytes, num_frames=5)
 
            if not frames:
                return jsonify({"error": "Could not extract any frames from the video"}), 400
 
            all_frame_slots = [run_yolo_on_frame(f) for f in frames if len(run_yolo_on_frame(f)) > 0]
            final_slots     = majority_vote(all_frame_slots)
 
        else:
            # ── IMAGE: decode → YOLO once ──
            if image_base64:
                frame = decode_image_base64(image_base64)
            elif image_url:
                img_bytes = download_bytes(image_url)
                np_arr    = np.frombuffer(img_bytes, np.uint8)
                frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                return jsonify({"error": "Provide 'image_base64' or 'image_url'"}), 400
 
            if frame is None:
                return jsonify({"error": "Failed to decode image. Ensure it is a valid JPG or PNG."}), 400
 
            final_slots = run_yolo_on_frame(frame)
 
        empty_count    = sum(1 for s in final_slots if s["status"] == "empty")
        occupied_count = sum(1 for s in final_slots if s["status"] == "occupied")
 
        return jsonify({
            "area_id":        area_id,
            "slots":          final_slots,
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