from flask import Flask, request, jsonify
import cv2
import numpy as np
import easyocr
import requests
import os
import uuid
from datetime import datetime
import threading

app = Flask(__name__)
reader = easyocr.Reader(['th', 'en'])

# Initialize directories
DETECTION_FOLDER = os.getenv('DETECTION_FOLDER', 'detections')
os.makedirs(DETECTION_FOLDER, exist_ok=True)

def preprocess_plate_image(img):
   """Preprocess license plate image for better OCR"""
   try:
       # Convert to grayscale
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
       # Apply adaptive thresholding
       blur = cv2.GaussianBlur(gray, (5, 5), 0)
       thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
       
       # Noise removal
       kernel = np.ones((3,3), np.uint8)
       opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
       
       # Resize for better OCR
       height, width = opening.shape
       if width > 300:
           scale = 300 / width
           new_width = 300
           new_height = int(height * scale)
           opening = cv2.resize(opening, (new_width, new_height))
           
       return opening
   except Exception as e:
       print(f"Error preprocessing plate image: {e}")
       return img

def read_license_plate(img):
   """Read license plate text using OCR"""
   try:
       # Preprocess image
       processed_img = preprocess_plate_image(img)
       
       # Run OCR
       results = reader.readtext(processed_img)
       
       if results:
           # Combine all detected text
           text = ' '.join([r[1] for r in results])
           confidence = sum([r[2] for r in results]) / len(results)
           
           # Clean the text (remove non-alphanumeric)
           text = ''.join(c for c in text if c.isalnum() or c.isspace())
           
           return text, confidence
       return 'Unknown', 0.0
       
   except Exception as e:
       print(f"Error reading license plate: {e}")
       return 'Unknown', 0.0

def save_violation_images(detection_id, motorcycle_img, plate_img):
   """Save violation images to disk"""
   try:
       motorcycle_path = os.path.join(DETECTION_FOLDER, f"{detection_id}_motorcycle.jpg")
       plate_path = os.path.join(DETECTION_FOLDER, f"{detection_id}_plate.jpg")
       
       cv2.imwrite(motorcycle_path, motorcycle_img)
       cv2.imwrite(plate_path, plate_img)
       
       return {
           'motorcycle_image': f"{detection_id}_motorcycle.jpg",
           'plate_image': f"{detection_id}_plate.jpg"
       }
   except Exception as e:
       print(f"Error saving violation images: {e}")
       return None

def process_violation(data):
   """Process violation data and save to database"""
   try:
       # Get paths
       motorcycle_path = os.path.join(DETECTION_FOLDER, data['motorcycle_image'])
       plate_path = os.path.join(DETECTION_FOLDER, data['plate_image'])
       
       # Read plate image
       plate_img = cv2.imread(plate_path)
       if plate_img is not None:
           # Read license plate
           plate_text, plate_confidence = read_license_plate(plate_img)
           
           # Prepare violation data
           violation_data = {
               'id': data['detection_id'],
               'video_name': data['filename'],
               'frame_number': data['frame_number'],
               'license_plate_text': plate_text,
               'license_plate_confidence': plate_confidence,
               'motorcycle_image': data['motorcycle_image'],
               'plate_image': data['plate_image'],
               'confidence': data['confidence'],
               'timestamp': datetime.now().isoformat()
           }
           
           # Save to database
           response = requests.post(
               'http://database:5003/violations',
               json=violation_data
           )
           
           if response.status_code != 200:
               print(f"Error saving to database: {response.text}")
           
           return violation_data
           
   except Exception as e:
       print(f"Error processing violation: {e}")
       return None

@app.route('/process_frame', methods=['POST'])
def process_frame():
   try:
       data = request.get_json()
       if not data:
           return jsonify({'error': 'No data received'}), 400
           
       # Process violation asynchronously
       threading.Thread(target=process_violation, args=(data,)).start()
       
       return jsonify({'success': True})
       
   except Exception as e:
       print(f"Error in process_frame: {e}")
       return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
   """Health check endpoint"""
   return jsonify({'status': 'healthy'})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5002)