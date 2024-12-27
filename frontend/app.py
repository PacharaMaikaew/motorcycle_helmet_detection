from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import cv2
import os
import time
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config.update(
    UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads'),
    MAX_CONTENT_LENGTH=2 * 1024 * 1024 * 1024  # 2GB max-size
)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return jsonify({'error': 'Video file not found'}), 404

        def generate_frames():
            try:
                # ลองรับ stream จาก detector service ก่อน
                response = requests.get(
                    'http://detector:5001/process',
                    params={
                        'video_path': video_path,
                        'filename': filename
                    },
                    stream=True,
                    timeout=5
                )
                
                if response.status_code == 200:
                    # ถ้าเชื่อมต่อกับ detector service ได้ ให้ส่ง stream จาก detector
                    for chunk in response.iter_content(chunk_size=1024):
                        yield chunk
                else:
                    # ถ้า detector service ไม่พร้อม ให้แสดงวิดีโอแบบปกติ
                    yield from stream_normal_video(video_path)
                    
            except requests.exceptions.RequestException as e:
                print(f"Detector service error: {e}, falling back to normal video")
                yield from stream_normal_video(video_path)
                
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    except Exception as e:
        print(f"Video feed error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def stream_normal_video(video_path):
    """Stream video without detection"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
        
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # ปรับขนาดและคุณภาพของเฟรม
            frame = cv2.resize(frame, (854, 480))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # ควบคุม frame rate
            time.sleep(1/30)
            
    except Exception as e:
        print(f"Error streaming video: {e}")
    finally:
        cap.release()

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:    
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)
        
        print(f"Video saved at: {video_path}")
        
        # เริ่มประมวลผลที่ detector service แบบ async
        try:
            requests.post(
                'http://detector:5001/process',
                json={
                    'video_path': video_path,
                    'filename': filename
                },
                timeout=1  # ลด timeout เพื่อไม่ให้ติดค้าง
            )
        except Exception as e:
            print(f"Warning: Detector service notification failed: {e}")
            # ไม่ต้อง return error เพราะยังสามารถแสดงวิดีโอได้
        
        return jsonify({
            'success': True,
            'filename': filename,
            'video_path': video_path
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)