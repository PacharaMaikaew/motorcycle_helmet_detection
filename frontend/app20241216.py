from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import requests
import os
from werkzeug.utils import secure_filename
import cv2
import threading
import time

app = Flask(__name__)

# Set maximum file size to 2GB
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024

# Add configuration for request
app.config.update(
    UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', 'uploads'),
    DETECTION_FOLDER=os.getenv('DETECTION_FOLDER', 'detections'),
    STREAM_FOLDER=os.getenv('STREAM_FOLDER', 'streams'),
)

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], 
              app.config['DETECTION_FOLDER'],
              app.config['STREAM_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_file_size', methods=['POST'])
def check_file_size():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    file.seek(0, os.SEEK_END)
    size = file.tell()
    
    if size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({
            'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"]/1024/1024/1024:.2f}GB'
        }), 413
    
    return jsonify({'success': True})

@app.route('/upload', methods=['POST'])
def upload_video():
    print("Upload request received")
    
    if 'video' not in request.files:
        print("No video file in request")
        return jsonify({'error': 'No video file provided'}), 400
        
    video = request.files['video']
    if video.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    try:    
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving video to: {video_path}")
        video.save(video_path)
        print(f"Video saved successfully: {filename}")

        # ส่งวิดีโอไปประมวลผลที่ detector service
        detector_response = requests.post(
            'http://detector:5001/process',
            json={'video_path': video_path, 'filename': filename}
        )
        
        return jsonify({
            'success': True,
            'filename': filename,
            'video_path': video_path
        })
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed/<filename>')
def video_feed(filename):
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return jsonify({'error': 'Video file not found'}), 404

        def generate_frames():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video file {video_path}")
                return
            
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            delay = 1 / frame_rate if frame_rate > 0 else 1 / 30

            try:
                while True:
                    success, frame = cap.read()
                    if not success:
                        print("End of video stream")
                        break

                    # Resize for streaming
                    frame = cv2.resize(frame, (854, 480))
                    
                    # Encode to JPEG
                    success, buffer = cv2.imencode('.jpg', frame)
                    if not success:
                        print("Error encoding frame")
                        continue

                    # Yield encoded frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    # Simulate frame rate
                    time.sleep(delay)

            except Exception as e:
                print(f"Error while generating frames: {e}")
            finally:
                cap.release()

        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    except Exception as e:
        print(f"Error in video_feed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/detections/<path:filename>')
def serve_detection(filename):
    return send_from_directory(app.config['DETECTION_FOLDER'], filename)

@app.route('/stop_processing/<filename>', methods=['POST'])
def stop_processing(filename):
    try:
        response = requests.post(
            'http://detector:5001/stop',
            json={'filename': filename}
        )
        return jsonify(response.json())
    except Exception as e:
        print(f"Error stopping processing: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)