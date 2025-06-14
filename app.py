from flask import Flask, render_template, Response, jsonify
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")

labels = ['1', '2', '3', '4',
          '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z']

cap = cv2.VideoCapture(0)

latest_pred = {"label": "", "confidence": 0.0}

def preprocess(image):
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
            class_index = np.argmax(preds)
            class_label = labels[class_index]
            confidence = np.max(preds)

            latest_pred = {"label": class_label, "confidence": confidence}

            # Send raw frame without text overlay
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
