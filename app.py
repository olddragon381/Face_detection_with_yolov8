from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import math

app = Flask(__name__)

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

def gen_frames():
    while True:
        success, frame = cap.read()  # Read the camera frame
        if not success:
            break

        # Perform inference on the image
        results = model(frame)

        # Process results
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # Class name (currently hardcoded as "face", modify as needed)
                cls = int(box.cls[0])
                class_name = "face"  # Replace with your actual class name mapping if different

                # Display object details
                org = (x1, y1 - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, f"{class_name} {confidence:.2f}", org, font, fontScale, color, thickness)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Use generator to yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Render the HTML page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False, port=5110)
