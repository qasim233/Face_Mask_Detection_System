import os
import base64
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

app = Flask(__name__)

# Paths to models
MODEL_WEIGHTS_PATH = 'face_mask_detection.h5'
PROTOTXT_PATH = 'deploy.prototxt'
CAFFE_MODEL_PATH = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
IMAGE_SIZE = 224
CONF_THRESHOLD = 0.5

# Build mask classification model architecture and load weights
base = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = False
inp = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base(inp, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
out = Dense(units=2, activation='softmax', dtype='float32')(x)
mask_model = Model(inp, out)
mask_model.load_weights(MODEL_WEIGHTS_PATH)

class_names = ['with_mask', 'without_mask']  # adjust order as in training

# Load DNN face detector
if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFE_MODEL_PATH):
    raise FileNotFoundError('DNN face detector files missing')
face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFE_MODEL_PATH)

# Preprocess face ROI
def preprocess_face(face_img):
    img = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    _, binar = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, kernel)
    mask = closed.astype(np.float32) / 255.0
    mask3 = np.stack([mask] * 3, axis=-1)
    masked = (rgb * mask3).astype(np.uint8)
    return tf.keras.applications.mobilenet_v2.preprocess_input(masked)

# Detect mask on face ROI
def detect_mask(face_img):
    x = preprocess_face(face_img)
    preds = mask_model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
    idx = np.argmax(preds)
    return class_names[idx], preds[idx]

# Generator for webcam frames
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process the frame and get statistics
        processed_frame, stats = process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# Process a frame and return both the processed frame and statistics
def process_frame(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    # Statistics
    total_faces = 0
    masked_faces = 0
    unmasked_faces = 0
    confidence_sum = 0
    
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < CONF_THRESHOLD:
            continue
            
        total_faces += 1
        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        face = frame[max(0, startY):min(h, endY), max(0, startX):min(w, endX)]
        if face.size == 0:
            continue
            
        label, prob = detect_mask(face)
        confidence_sum += prob * 100
        
        if label == 'with_mask':
            masked_faces += 1
        else:
            unmasked_faces += 1
            
        color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)
        text = f"{label}: {prob*100:.1f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Calculate average confidence
    avg_confidence = confidence_sum / total_faces if total_faces > 0 else 0
    
    stats = {
        'total_faces': total_faces,
        'masked_faces': masked_faces,
        'unmasked_faces': unmasked_faces,
        'confidence': avg_confidence
    }
    
    return frame, stats

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video', endpoint='video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read and process the image
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Process the image and get statistics
    processed_img, stats = process_frame(img)
    
    # Convert the processed image to base64 for embedding in JSON
    ret, buffer = cv2.imencode('.jpg', processed_img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    # Return both the image and statistics as JSON
    response = {
        'image_base64': f'data:image/jpeg;base64,{img_str}',
        'total_faces': stats['total_faces'],
        'masked_faces': stats['masked_faces'],
        'unmasked_faces': stats['unmasked_faces'],
        'confidence': stats['confidence']
    }
    
    return jsonify(response)

@app.route('/stats', methods=['POST'])
def get_stats():
    """Endpoint to get just the statistics without the image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read and process the image
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Process the image and get statistics
    _, stats = process_frame(img)
    
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
