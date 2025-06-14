from flask import Flask, render_template, Response, jsonify
import cv2
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Path to dataset directory used during training
dataset_dir = r"D:/Sign_Language_Detector/Indian"

# Automatically load sorted folder names as labels
labels = sorted(os.listdir(dataset_dir))
print("Labels detected:", labels)  # DEBUG: check this in console

model = tf.keras.models.load_model("model.h5")

cap = cv2.VideoCapture(0)

latest_pred = {"label": "", "confidence": 0.0}

def preprocess(image):
    # Convert BGR (OpenCV) to RGB (model training)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def gen_frames():
    global latest_pred
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            input_image = preprocess(frame)
            preds = model.predict(input_image, verbose=0)

            # DEBUG: print raw predictions
            # print(preds)

            class_index = np.argmax(preds)
            class_label = labels[class_index]
            confidence = np.max(preds)

            latest_pred = {"label": class_label, "confidence": confidence}

            # Send raw frame without overlay
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    return jsonify({
        "label": latest_pred["label"],
        "confidence": f"{latest_pred['confidence']*100:.1f}%"
    })

import atexit
@atexit.register
def cleanup():
    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
