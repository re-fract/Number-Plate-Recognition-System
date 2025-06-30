# License Plate Recognition System (ANPR)

![System Overview](https://github.com/Tkvmaster/ANPR-System/blob/main/utility_files/anpr_logo.jpg?raw=true)

A comprehensive Automatic Number Plate Recognition (ANPR) system with real-time video processing, live webcam detection, and image analysis capabilities.

## Features

- 🎥 **Real-time Video Processing**: Upload videos for frame-by-frame license plate detection
- 📷 **Live Webcam Feed**: Detect license plates from your webcam in real-time
- 🖼️ **Image Analysis**: Upload single images for plate detection
- 🚀 **High Performance**: Utilizes YOLO object detection and EasyOCR
- 📊 **Detailed Analytics**: Tracks detection confidence and OCR accuracy
- 🔌 **WebSocket Integration**: Real-time progress updates during video processing

## System Architecture

```
Frontend (React) ↔ Backend (Flask) ↔ AI Models (YOLO + EasyOCR)
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- NVIDIA GPU (recommended for optimal performance)
- CUDA and cuDNN (for GPU acceleration)

## Installation

### Backend Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLO model weights and place them in `runs/detect/train/weights/`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

## Running the System

### Start the Backend Server
```bash
python app.py
```

### Start the Frontend Development Server
```bash
cd frontend
npm start
```

The application will be available at `http://localhost:3000`

## Configuration

### Environment Variables

Create a `.env` file in the backend directory with the following variables:

```
FLASK_ENV=development
SECRET_KEY=your-secret-key
PORT=5000
```

### Model Paths

Ensure the model paths in `predict_vid.py` and `app.py` point to your YOLO model weights.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload_image` | POST | Process a single image for license plate detection |
| `/upload_video_realtime` | POST | Start real-time video processing |
| `/start_webcam` | POST | Activate webcam feed |
| `/stop_webcam` | POST | Deactivate webcam feed |
| `/video_feed` | GET | Webcam video stream |
| `/health` | GET | System health check |

## WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `video_info` | Server → Client | Video metadata (fps, resolution) |
| `frame_processed` | Server → Client | Processed frame with detection results |
| `processing_complete` | Server → Client | Video processing completion |
| `processing_error` | Server → Client | Processing error notification |
| `join_session` | Client → Server | Join a processing session |
| `stop_processing` | Client → Server | Stop active processing session |

## License

This project is licensed under the MIT License.

## Acknowledgments

- YOLOv8 for object detection
- EasyOCR for text recognition
- SORT for object tracking
- Flask and React for web framework
