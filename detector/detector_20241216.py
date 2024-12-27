from flask import Flask, request, jsonify, Response
from ultralytics import YOLO
import cv2
import torch
import os
import redis
import json
import numpy as np
import requests
import threading
import time
from threading import Event, Lock
from queue import Queue

app = Flask(__name__)
model = None
redis_client = redis.Redis(host='redis', port=6379)

video_processes = {}
process_locks = {}
processing_status = {}
video_queues = {}
video_threads = {}
MAX_QUEUE_SIZE = 10

def load_model():
    global model
    model_path = os.getenv('MODEL_PATH')
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = YOLO(model_path)

def send_to_processor(filename, frame_number, detections):
    try:
        requests.post(
            'http://processor:5002/process_frame',
            json={
                'filename': filename,
                'frame_number': frame_number,
                'detections': detections
            }
        )
    except Exception as e:
        print(f"Error sending to processor: {e}")

@app.route('/stop', methods=['POST'])
def stop_processing():
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400

        with process_locks.get(filename, Lock()):
            if filename in processing_status:
                processing_status[filename] = False
                if filename in video_processes:
                    video_processes[filename].set()
                return jsonify({'success': True})
            return jsonify({'error': 'No active processing found'}), 404

    except Exception as e:
        print(f"Error in stop_processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['GET', 'POST'])
def process_video():
    try:
        if request.method == 'GET':
            video_path = request.args.get('video_path')
            filename = request.args.get('filename')
        else:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data received'}), 400
            video_path = data.get('video_path')
            filename = data.get('filename')

        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': f'Video file not found: {video_path}'}), 400

        def generate_frames():
            while processing_status.get(filename, True) or not video_queues[filename].empty():
                try:
                    frame_data = video_queues[filename].get(timeout=0.1)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                except:
                    continue

        if filename not in video_queues:
            video_queues[filename] = Queue(maxsize=MAX_QUEUE_SIZE)
            processing_status[filename] = True

            def process_video_frames():
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                frame_count = 0
                last_frame_time = time.time()
                
                try:
                    while cap.isOpened() and processing_status.get(filename, True):
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # ปรับขนาดและประมวลผลเฟรม
                        frame = cv2.resize(frame, (640, 360))
                        
                        with torch.inference_mode():
                            results = model(frame, conf=0.4, iou=0.5, max_det=50, agnostic_nms=True)[0]
                        
                        if len(results.boxes) > 0:
                            boxes = results.boxes
                            xyxy = boxes.xyxy.cpu().numpy()
                            cls = boxes.cls.cpu().numpy()
                            conf = boxes.conf.cpu().numpy()
                            
                            for i, box in enumerate(xyxy):
                                x1, y1, x2, y2 = map(int, box)
                                class_id = int(cls[i])
                                confidence = conf[i]
                                
                                # กำหนดสีและป้ายกำกับ
                                colors = {
                                    0: ((255, 165, 0), 'Motorcycle'),  # สีส้ม
                                    1: ((0, 255, 0), 'Helmet'),       # สีเขียว
                                    2: ((255, 0, 0), 'Head'),         # สีแดง
                                    3: ((0, 0, 255), 'No Helmet')     # สีน้ำเงิน
                                }
                                color, label = colors.get(class_id, ((128, 128, 128), 'Unknown'))
                                
                                # วาดกรอบและข้อความ
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                label_text = f'{label} {confidence:.2f}'
                                cv2.putText(frame, label_text, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                                # ส่งข้อมูลเมื่อพบการละเมิด
                                if class_id == 3:  # No Helmet
                                    threading.Thread(
                                        target=send_to_processor,
                                        args=(filename, frame_count, results.boxes.data.tolist()),
                                        daemon=True
                                    ).start()

                        # เข้ารหัสและส่งเฟรม
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_data = buffer.tobytes()
                        
                        try:
                            if not video_queues[filename].full():
                                video_queues[filename].put_nowait(frame_data)
                            else:
                                video_queues[filename].get_nowait()
                                video_queues[filename].put_nowait(frame_data)
                        except:
                            continue

                        # ควบคุม FPS
                        frame_count += 1
                        elapsed_time = time.time() - last_frame_time
                        if elapsed_time < 1/30:
                            time.sleep(1/30 - elapsed_time)
                        last_frame_time = time.time()

                        # ล้าง GPU memory
                        if frame_count % 30 == 0:
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    print(f"Error in frame processing: {e}")
                finally:
                    cap.release()
                    processing_status[filename] = False

            # เริ่มการประมวลผลในเทรดใหม่
            process_thread = threading.Thread(target=process_video_frames, daemon=True)
            video_threads[filename] = process_thread
            process_thread.start()

        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    except Exception as e:
        print(f"Error in process_video: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        load_model()
        app.run(host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"Error starting detector service: {e}")