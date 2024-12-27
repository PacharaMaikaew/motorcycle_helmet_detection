from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import requests
import os
from werkzeug.utils import secure_filename
import cv2
import threading
import time

app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', 'uploads'),
    DETECTION_FOLDER=os.getenv('DETECTION_FOLDER', 'detections'),
    STREAM_FOLDER=os.getenv('STREAM_FOLDER', 'streams'),
    MAX_CONTENT_LENGTH=1024 * 1024 * 1024  # 1GB max file size
)

active_streams = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)
    
    # Start processing in background thread
    thread = threading.Thread(target=process_video, args=(video_path, filename))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'filename': filename
    })

def process_video(video_path, filename):
    try:
        response = requests.post(
            'http://detector:5001/process',
            json={'video_path': video_path, 'filename': filename}
        )
        
        if response.status_code != 200:
            print(f"Error processing video: {response.text}")
    except Exception as e:
        print(f"Error in process_video: {e}")

@app.route('/video_feed/')
def video_feed(filename):
    def generate_frames():
        stream_path = os.path.join(app.config['STREAM_FOLDER'], filename)
        while True:
            if os.path.exists(stream_path):
                frame = cv2.imread(stream_path)
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)  # ~30 FPS

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/violations/')
def get_violations(filename):
    response = requests.get(
        f'http://database:5003/violations/{filename}'
    )
    return jsonify(response.json())

@app.route('/detections/')
def serve_detection(filename):
    return send_from_directory(app.config['DETECTION_FOLDER'], filename)

if __name__ == '__main__':
    for folder in [app.config['UPLOAD_FOLDER'], 
                  app.config['DETECTION_FOLDER'],
                  app.config['STREAM_FOLDER']]:
        os.makedirs(folder, exist_ok=True)
    app.run(host='0.0.0.0', port=5000)