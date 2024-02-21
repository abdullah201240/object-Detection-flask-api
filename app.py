from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import math
import os
from flask_cors import CORS, cross_origin
from datetime import datetime
import logging

app = Flask(__name__)
cors = CORS(app, resources={r"/detect_objects": {"origins": "http://192.168.0.110"}})
print("Received POST request to /detect_objects")

# Load YOLO model
YOLO_MODEL_PATH = "yolov8n.pt"
model = YOLO(YOLO_MODEL_PATH)

# Define class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


@app.route('/detect_objects', methods=['POST', 'OPTIONS'])
@cross_origin()
def detect_objects():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']

        temp_path = "temp_image.jpg"
        image_file.save(temp_path)

        results = model(temp_path, stream=True)

        img = cv2.imread(temp_path)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(img, (int(x1), int(y1)),
                              (int(x2), int(y2)), (255, 0, 255), 3)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'

                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = (int(x1 + t_size[0]), int(y1 - t_size[1] - 3))

                cv2.rectangle(img, (int(x1), int(y1)), c2, [
                              255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (int(x1), int(y1 - 2)), 0, 1,
                            [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_name = f"annotated_{timestamp}_temp_image.jpg"
        base_directory = "/Users/abdullahalsakib/Downloads/Trimester13/React Native/image/"
        path = os.path.join(base_directory, image_name)

        cv2.imwrite(path, img)

        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logging.exception(f"Error while removing temporary file: {str(e)}")

        return send_file(path, mimetype='image/jpeg'), 200

    except Exception as e:
        logging.exception("An error occurred during object detection:")
        return jsonify({'error': 'An unexpected error occurred'}), 500


if __name__ == '__main__':
    print("Flask app is running. Access at http://192.168.0.110:5000")
    app.run(host='192.168.0.110', port=5000, debug=False)
