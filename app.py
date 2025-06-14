from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Define your labels (same order as the training folders)
labels = ['1', '2', '3', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
          'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
          'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize webcam
cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    # Preprocess frame for model
    img = cv2.resize(frame, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Flip the frame (mirror effect)
            frame = cv2.flip(frame, 1)

            # Preprocess and predict
            img = preprocess_frame(frame)
            prediction = model.predict(img)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            class_name = labels[class_index]

            # Show prediction on frame
            label_text = f"{class_name} ({confidence*100:.1f}%)"
            cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame to browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
