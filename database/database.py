from flask import Flask, request, jsonify
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
DB_PATH = os.getenv('DB_PATH', 'violations.db')

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id TEXT PRIMARY KEY,
                video_name TEXT,
                frame_number INTEGER,
                timestamp DATETIME,
                license_plate_text TEXT,
                license_plate_confidence FLOAT,
                motorcycle_image TEXT,
                plate_image TEXT,
                confidence FLOAT
            )
        ''')

@app.route('/violations', methods=['POST'])
def add_violation():
    data = request.json
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                '''INSERT INTO violations 
                   (id, video_name, frame_number, timestamp,
                    license_plate_text, license_plate_confidence,
                    motorcycle_image, plate_image, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    data['id'],
                    data['video_name'],
                    data['frame_number'],
                    datetime.now(),
                    data.get('license_plate_text', 'Unknown'),
                    data.get('license_plate_confidence', 0.0),
                    data.get('motorcycle_image', ''),
                    data.get('plate_image', ''),
                    data.get('confidence', 0.0)
                )
            )
            return jsonify({'success': True})
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({'error': str(e)}), 500
    except KeyError as e:
        print(f"Missing required field: {e}")
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400

@app.route('/violations/<filename>')
def get_violations(filename):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''SELECT * FROM violations 
                   WHERE video_name = ? 
                   ORDER BY frame_number DESC''',
                (filename,)
            )
            violations = [dict(row) for row in cursor.fetchall()]
            
            if not violations:
                return jsonify([])  # Return empty array if no violations
                
            return jsonify(violations)
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # สร้างโฟลเดอร์และฐานข้อมูลก่อนเริ่ม server
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db()
    app.run(host='0.0.0.0', port=5003)