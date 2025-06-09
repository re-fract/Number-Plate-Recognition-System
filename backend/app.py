from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
import cv2
import os
from predict_vid import PlateDetector
from flask import send_from_directory
import easyocr
import numpy as np
from ultralytics import YOLO
import base64
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Initialize models
try:
    model = YOLO('../runs/detect/train/weights/best.pt')
    reader = easyocr.Reader(['en'])
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Failed to load models: {e}")

# Store active processing sessions
active_sessions = {}

# Global webcam state
webcam_active = False
webcam_detector = None
webcam_thread = None

def preprocess_plate(plate_roi):
    height, width = plate_roi.shape[:2]
    if height < 50 or width < 100:
        scale = max(100 / width, 50 / height) * 1.5
        plate_roi = cv2.resize(plate_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def process_video_realtime(video_path, session_id):
    """Process video and stream frames in real-time"""
    try:
        print(f"🎬 Starting real-time processing for session: {session_id}")
        print(f"📁 Video path: {video_path}")

        # Check if file exists
        if not os.path.exists(video_path):
            print(f"❌ Video file does not exist: {video_path}")
            socketio.emit('processing_error', {'error': 'Video file not found'}, room=session_id)
            return

        detector = PlateDetector(video_source=video_path)
        cap = detector.cap

        if not cap.isOpened():
            print("❌ Failed to open video capture")
            socketio.emit('processing_error', {'error': 'Could not open video'}, room=session_id)
            return

        # Get video properties with better error handling
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📊 Video properties:")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps}")
        print(f"   Resolution: {width}x{height}")

        # Handle cases where frame count is unreliable
        if total_frames <= 0:
            print("⚠️ Total frames is 0 or negative, counting manually...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print(f"📊 Manual count: {total_frames} frames")

        # Set default FPS if invalid
        if fps <= 0:
            fps = 25.0
            print(f"⚠️ Invalid FPS, defaulting to {fps}")

        # Emit initial video info to frontend
        socketio.emit('video_info', {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height
        }, room=session_id)

        frame_count = 0
        start_time = time.time()

        print(f"🚀 Starting frame processing...")

        while cap.isOpened() and session_id in active_sessions:
            ret, frame = cap.read()
            if not ret:
                print(f"📹 End of video reached at frame {frame_count}")
                break

            # Process frame with license plate detection
            processed_frame = detector.process_frame(frame)

            if processed_frame is not None and processed_frame.size > 0:
                # Resize for web display
                display_height, display_width = processed_frame.shape[:2]
                if display_width > 800:
                    scale = 800 / display_width
                    new_width = int(display_width * scale)
                    new_height = int(display_height * scale)
                    processed_frame = cv2.resize(processed_frame, (new_width, new_height))

                # Encode frame as JPEG
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                _, buffer = cv2.imencode('.jpg', processed_frame, encode_params)

                # Convert to base64
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # Calculate progress
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0

                # Emit frame to client
                socketio.emit('frame_processed', {
                    'frame': f"data:image/jpeg;base64,{frame_base64}",
                    'frame_number': frame_count + 1,
                    'total_frames': total_frames,
                    'progress': min(progress, 100),
                    'fps': fps
                }, room=session_id)

                frame_count += 1

                # Log progress
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    processing_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"🟢 Frame {frame_count}/{total_frames} ({progress:.1f}%) - Speed: {processing_fps:.1f} FPS")

                # Control processing speed
                time.sleep(0.05)
            else:
                print(f"⚠️ Frame {frame_count} processing failed")
                frame_count += 1

        cap.release()

        # Notify completion
        socketio.emit('processing_complete', {
            'total_frames': frame_count,
            'session_id': session_id,
            'message': f'Successfully processed {frame_count} frames'
        }, room=session_id)

        print(f"✅ Real-time processing complete: {frame_count} frames processed")

        # Clean up
        if session_id in active_sessions:
            del active_sessions[session_id]

        # Clean up uploaded file
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"🗑️ Cleaned up uploaded file: {video_path}")
        except Exception as e:
            print(f"⚠️ Failed to clean up file: {e}")

    except Exception as e:
        print(f"💥 Real-time processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        socketio.emit('processing_error', {'error': str(e)}, room=session_id)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        print("🖼️ Image upload request received")

        if 'image' not in request.files:
            return jsonify({"error": "No image file found"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        print(f"📸 Processing image: {file.filename}")

        # Read image directly from memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not decode image"}), 400

        print(f"✅ Image loaded: {image.shape}")

        # Run YOLO detection
        results = model(image)
        result = results[0]

        detections_found = 0

        # Process each detected license plate
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            print(f"🎯 Detection: confidence={conf:.2f}, bbox=({x1},{y1},{x2},{y2})")

            # Extract plate region
            plate_roi = image[y1:y2, x1:x2]

            if plate_roi.size == 0:
                continue

            # Preprocess and run OCR
            try:
                preprocessed = preprocess_plate(plate_roi)
                ocr_results = reader.readtext(preprocessed)

                # Get OCR text
                ocr_text = "Unknown"
                if ocr_results and len(ocr_results) > 0:
                    ocr_text = ocr_results[0][1]
                    print(f"📝 OCR Result: {ocr_text}")

                # Draw bounding box and text
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(image, f"{ocr_text} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

                detections_found += 1

            except Exception as e:
                print(f"⚠️ OCR processing failed: {e}")
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(image, f"Plate ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        print(f"🎉 Processing complete: {detections_found} plates detected")

        # Encode image as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        _, buffer = cv2.imencode('.jpg', image, encode_params)

        return Response(buffer.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        print(f"💥 Image processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/upload_video_realtime', methods=['POST'])
def upload_video_realtime():
    try:
        print("🎬 Real-time video upload request received")
        from datetime import datetime

        if 'video' not in request.files:
            print("❌ No video file in request")
            return jsonify({"error": "No video file found"}), 400

        file = request.files['video']
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({"error": "No file selected"}), 400

        # Generate session ID
        session_id = f"session_{datetime.now().timestamp()}"
        print(f"📋 Generated session ID: {session_id}")

        # Save uploaded file
        filename = file.filename
        video_path = os.path.join('uploads', f"{session_id}_{filename}")
        file.save(video_path)
        print(f"💾 Video saved to: {video_path}")

        # Verify file was saved
        if not os.path.exists(video_path):
            print("❌ Failed to save video file")
            return jsonify({"error": "Failed to save video file"}), 500

        file_size = os.path.getsize(video_path)
        print(f"✅ File saved successfully, size: {file_size} bytes")

        # Store session info
        active_sessions[session_id] = {
            'video_path': video_path,
            'status': 'processing'
        }

        # Start processing in background thread
        processing_thread = threading.Thread(
            target=process_video_realtime,
            args=(video_path, session_id)
        )
        processing_thread.daemon = True
        processing_thread.start()
        print(f"🚀 Started processing thread for session: {session_id}")

        return jsonify({
            "session_id": session_id,
            "status": "processing_started"
        }), 200

    except Exception as e:
        print(f"💥 Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def webcam_stream_generator():
    """Generator for webcam stream"""
    global webcam_detector, webcam_active

    try:
        frame_count = 0
        while webcam_active and webcam_detector and webcam_detector.cap.isOpened():
            ret, frame = webcam_detector.cap.read()
            if not ret:
                print("⚠️ Failed to read webcam frame")
                break

            try:
                processed = webcam_detector.process_frame(frame)
                if processed is None or processed.size == 0:
                    processed = frame

                success, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not success:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"📹 Streamed {frame_count} webcam frames")

            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                continue

    except Exception as e:
        print(f"💥 Webcam stream error: {str(e)}")
    finally:
        print("🔴 Webcam stream generator ended")

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_active, webcam_detector

    try:
        print("🎥 Starting webcam...")

        if webcam_active:
            return jsonify({"message": "Webcam already active"}), 200

        webcam_detector = PlateDetector(video_source=0)

        if not webcam_detector.cap.isOpened():
            print("❌ Failed to open webcam")
            return jsonify({"error": "Failed to access webcam. Make sure it's not being used by another application."}), 500

        webcam_active = True
        print("✅ Webcam started successfully")

        return jsonify({"message": "Webcam started successfully"}), 200

    except Exception as e:
        print(f"💥 Webcam start error: {str(e)}")
        webcam_active = False
        return jsonify({"error": str(e)}), 500

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_active, webcam_detector

    try:
        print("⏹️ Stopping webcam...")

        webcam_active = False

        if webcam_detector and webcam_detector.cap:
            webcam_detector.cap.release()
            print("📷 Webcam released")

        webcam_detector = None
        print("✅ Webcam stopped successfully")

        return jsonify({"message": "Webcam stopped successfully"}), 200

    except Exception as e:
        print(f"💥 Webcam stop error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/webcam_status')
def webcam_status():
    """Check if webcam is currently active"""
    global webcam_active
    return jsonify({
        "active": webcam_active,
        "detector_ready": webcam_detector is not None
    })

# KEEP ONLY THIS video_feed FUNCTION (remove the duplicate one)
@app.route('/video_feed')
def video_feed():
    global webcam_active

    if not webcam_active:
        # Return a placeholder response when webcam is not active
        def empty_stream():
            yield b''

        return Response(empty_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return Response(webcam_stream_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Socket.IO events
@socketio.on('join_session')
def on_join_session(data):
    session_id = data['session_id']
    join_room(session_id)
    print(f"🔗 Client joined session: {session_id}")

@socketio.on('stop_processing')
def on_stop_processing(data):
    session_id = data['session_id']
    if session_id in active_sessions:
        del active_sessions[session_id]
        print(f"⏹️ Processing stopped for session: {session_id}")

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'webcam_available': cv2.VideoCapture(0).isOpened(),
        'models_loaded': 'model' in globals() and 'reader' in globals()
    })

if __name__ == '__main__':
    print("🚀 Starting Flask-SocketIO server...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)