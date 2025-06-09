import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import './App.css';

function App() {
  const [mode, setMode] = useState('video');
  const [videoFile, setVideoFile] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [videoSrc, setVideoSrc] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState(0);
  const [isDragOver, setIsDragOver] = useState(false);
  const [webcamActive, setWebcamActive] = useState(false);
  const [webcamError, setWebcamError] = useState('');

  // Real-time streaming states
  const [isLiveProcessing, setIsLiveProcessing] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [socket, setSocket] = useState(null);
  const [frameInfo, setFrameInfo] = useState({ current: 0, total: 0, fps: 0 });
  const [videoInfo, setVideoInfo] = useState({ total: 0, fps: 0, width: 0, height: 0 });

  // Initialize Socket.IO
  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('video_info', (data) => {
      console.log('Video info received:', data);
      setVideoInfo({
        total: data.total_frames,
        fps: data.fps,
        width: data.width,
        height: data.height
      });
      setFrameInfo({
        current: 0,
        total: data.total_frames,
        fps: data.fps
      });
    });

    newSocket.on('frame_processed', (data) => {
      setCurrentFrame(data.frame);
      setProgress(data.progress);
      setFrameInfo({
        current: data.frame_number,
        total: data.total_frames,
        fps: data.fps
      });
    });

    newSocket.on('processing_complete', (data) => {
      console.log('Processing complete:', data);
      setIsLiveProcessing(false);
      setProcessing(false);
      setProgress(100);
    });

    newSocket.on('processing_error', (data) => {
      console.error('Processing error:', data);
      setError(data.error);
      setIsLiveProcessing(false);
      setProcessing(false);
    });

    return () => newSocket.close();
  }, []);

  const modes = [
    { id: 'video', label: '📹 Video Upload', icon: '🎬' },
    { id: 'webcam', label: '📷 Live Webcam', icon: '📡' },
    { id: 'image', label: '🖼️ Image Upload', icon: '📸' }
  ];

  // Webcam control functions
  const startWebcam = async () => {
    try {
      setWebcamError('');
      const response = await fetch('http://localhost:5000/start_webcam', {
        method: 'POST'
      });

      if (response.ok) {
        setWebcamActive(true);
        console.log('✅ Webcam started');
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to start webcam');
      }
    } catch (err) {
      setWebcamError('Failed to start webcam: ' + err.message);
      console.error('❌ Webcam start error:', err);
    }
  };

  const stopWebcam = async () => {
    try {
      const response = await fetch('http://localhost:5000/stop_webcam', {
        method: 'POST'
      });

      if (response.ok) {
        setWebcamActive(false);
        console.log('⏹️ Webcam stopped');
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to stop webcam');
      }
    } catch (err) {
      setWebcamError('Failed to stop webcam: ' + err.message);
      console.error('❌ Webcam stop error:', err);
    }
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setError('');
    setVideoFile(null);
    setImageFile(null);
    setVideoSrc(null);
    setImageSrc(null);
    setCurrentFrame(null);
    setIsLiveProcessing(false);
    setProcessing(false);
    setProgress(0);

    // Reset webcam state when switching modes
    if (newMode !== 'webcam' && webcamActive) {
      stopWebcam();
    }
    setWebcamError('');
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (mode === 'video') {
        setVideoFile(file);
      } else if (mode === 'image') {
        setImageFile(file);
      }
      setError('');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];

    if (mode === 'video' && file && file.type.startsWith('video/')) {
      setVideoFile(file);
      setError('');
    } else if (mode === 'image' && file && file.type.startsWith('image/')) {
      setImageFile(file);
      setError('');
    } else {
      setError(`Please drop a valid ${mode} file`);
    }
  };

  const handleVideoUploadRealtime = async () => {
    if (!videoFile) {
      setError('Please select a video file');
      return;
    }

    setProcessing(true);
    setIsLiveProcessing(true);
    setError('');
    setCurrentFrame(null);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("video", videoFile);

      const res = await fetch("http://localhost:5000/upload_video_realtime", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      const newSessionId = data.session_id;
      setSessionId(newSessionId);

      socket.emit('join_session', { session_id: newSessionId });

    } catch (err) {
      setError(err.message);
      setProcessing(false);
      setIsLiveProcessing(false);
    }
  };

  const handleImageUpload = async () => {
    if (!imageFile) {
      setError('Please select an image file');
      return;
    }

    setProcessing(true);
    setError('');
    setImageSrc(null);

    try {
      const formData = new FormData();
      formData.append("image", imageFile);

      const res = await fetch("http://localhost:5000/upload_image", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Image upload failed');
      }

      const blob = await res.blob();
      const imageUrl = URL.createObjectURL(blob);
      setImageSrc(imageUrl);

    } catch (err) {
      setError(err.message);
      console.error('Image upload error:', err);
    } finally {
      setProcessing(false);
    }
  };

  const stopProcessing = () => {
    if (sessionId && socket) {
      socket.emit('stop_processing', { session_id: sessionId });
      setIsLiveProcessing(false);
      setProcessing(false);
      setCurrentFrame(null);
    }
  };

  const clearFiles = () => {
    setVideoFile(null);
    setImageFile(null);
    setVideoSrc(null);
    setImageSrc(null);
    setCurrentFrame(null);
    setError('');
    setIsLiveProcessing(false);
    setProgress(0);
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>🚗 License Plate Recognition System</h1>
          <p>Advanced AI-powered detection and OCR for license plates</p>
        </header>

        {/* Mode Selection */}
        <div className="mode-selector">
          {modes.map((modeOption) => (
            <button
              key={modeOption.id}
              className={`mode-btn ${mode === modeOption.id ? 'active' : ''}`}
              onClick={() => handleModeChange(modeOption.id)}
            >
              <span className="mode-icon">{modeOption.icon}</span>
              {modeOption.label}
            </button>
          ))}
        </div>

        {/* Video Upload Mode */}
        {mode === 'video' && (
          <div className="upload-section">
            <h2>📹 Real-time Video Processing</h2>

            {!isLiveProcessing && (
              <div
                className={`drop-zone ${isDragOver ? 'drag-over' : ''} ${videoFile ? 'has-file' : ''}`}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onDragEnter={() => setIsDragOver(true)}
                onDragLeave={() => setIsDragOver(false)}
              >
                {!videoFile ? (
                  <>
                    <div className="upload-icon">🎬</div>
                    <h3>Drag & Drop Video Here</h3>
                    <p>or click to browse files</p>
                    <input
                      type="file"
                      accept="video/*"
                      onChange={handleFileChange}
                      className="file-input"
                      disabled={processing}
                    />
                  </>
                ) : (
                  <div className="file-info">
                    <div className="file-icon">🎬</div>
                    <div className="file-details">
                      <h4>{videoFile.name}</h4>
                      <p>{(videoFile.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                    <button onClick={clearFiles} className="clear-btn">✕</button>
                  </div>
                )}
              </div>
            )}

            {videoFile && !isLiveProcessing && (
              <div className="action-buttons">
                <button
                  onClick={handleVideoUploadRealtime}
                  disabled={processing}
                  className={`upload-btn ${processing ? 'processing' : ''}`}
                >
                  🚀 Start Live Processing
                </button>
              </div>
            )}

            {/* Real-time Processing Display */}
            {isLiveProcessing && (
              <div className="live-processing-section">
               <div className="processing-header">
                <h3>🔴 Live Processing</h3>
                <button onClick={stopProcessing} className="stop-btn">
                  <span>⏹</span>
                  Stop Processing
                </button>
              </div>

                {videoInfo.total > 0 && (
                  <div className="video-info-display">
                    <p>📊 Video: {videoInfo.width}x{videoInfo.height} @ {videoInfo.fps.toFixed(1)} FPS</p>
                    <p>📹 Total Frames: {videoInfo.total}</p>
                  </div>
                )}

                {currentFrame && (
                  <div className="live-frame-container">
                    <img
                      src={currentFrame}
                      alt="Live processing"
                      className="live-frame"
                    />
                  </div>
                )}

                <div className="processing-stats">
                  <div className="progress-section">
                    <div className="progress-bar">
                      <div className="progress-fill" style={{width: `${progress}%`}}></div>
                    </div>
                    <p className="progress-text">
                      {Math.round(progress)}% - Frame {frameInfo.current}/{frameInfo.total}
                      {frameInfo.fps > 0 && ` @ ${frameInfo.fps.toFixed(1)} FPS`}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Live Webcam Mode */}
        {mode === 'webcam' && (
          <div className="webcam-section">
            <h2>📷 Live Webcam Feed</h2>

            {/* Webcam Controls */}
            <div className="webcam-controls">
              {!webcamActive ? (
                <button
                  onClick={startWebcam}
                  className="start-btn"
                  disabled={processing}
                >
                  Start Webcam
                </button>
              ) : (
                <button
                  onClick={stopWebcam}
                  className="stop-webcam-btn"
                >
                  Stop Webcam
                </button>
              )}

              <div className="webcam-status">
                {webcamActive && (
                  <span className="status-indicator active">
                    🟢 Live - Recording
                  </span>
                )}
                {!webcamActive && !webcamError && (
                  <span className="status-indicator inactive">
                    ⚫ Webcam Stopped
                  </span>
                )}
              </div>
            </div>

            {/* Webcam Feed Display */}
            <div className="webcam-container">
              {webcamActive ? (
                <img
                  src={`http://localhost:5000/video_feed?t=${Date.now()}`}
                  alt="Live webcam feed"
                  className="webcam-feed"
                  onError={(e) => {
                    console.error('Webcam feed error');
                    setWebcamError('Lost connection to webcam feed');
                  }}
                  onLoad={() => {
                    setWebcamError(''); // Clear error when feed loads successfully
                  }}
                />
              ) : (
                <div className="webcam-placeholder">
                  <div className="placeholder-content">
                    <span className="placeholder-icon">📷</span>
                    <h3>Webcam Feed</h3>
                    <p>Click "Start Webcam" to begin live license plate detection</p>
                  </div>
                </div>
              )}
            </div>

            {webcamActive && (
              <p className="webcam-info">
                🎯 Real-time license plate detection active
              </p>
            )}

            {webcamError && (
              <div className="webcam-error">
                <span className="error-icon">⚠️</span>
                {webcamError}
              </div>
            )}
          </div>
        )}

        {/* Image Upload Mode */}
        {mode === 'image' && (
          <div className="upload-section">
            <h2>🖼️ Image Processing</h2>
            <div
              className={`drop-zone ${isDragOver ? 'drag-over' : ''} ${imageFile ? 'has-file' : ''}`}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onDragEnter={() => setIsDragOver(true)}
              onDragLeave={() => setIsDragOver(false)}
            >
              {!imageFile ? (
                <>
                  <div className="upload-icon">📸</div>
                  <h3>Drag & Drop Image Here</h3>
                  <p>or click to browse files (JPG, PNG, etc.)</p>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="file-input"
                    disabled={processing}
                  />
                </>
              ) : (
                <div className="file-info">
                  <div className="file-icon">📸</div>
                  <div className="file-details">
                    <h4>{imageFile.name}</h4>
                    <p>{(imageFile.size / 1024).toFixed(2)} KB</p>
                  </div>
                  <button onClick={clearFiles} className="clear-btn">✕</button>
                </div>
              )}
            </div>

            {imageFile && (
              <div className="action-buttons">
                <button
                  onClick={handleImageUpload}
                  disabled={processing}
                  className={`upload-btn ${processing ? 'processing' : ''}`}
                >
                  {processing ? (
                    <>
                      <span className="spinner"></span>
                      Processing Image...
                    </>
                  ) : (
                    <>🔍 Analyze Image</>
                  )}
                </button>
              </div>
            )}

            {imageSrc && (
              <div className="results-section">
                <h3>📊 Detection Results</h3>
                <div className="image-container">
                  <img src={imageSrc} alt="Processed result" className="result-image" />
                </div>
                <p className="result-info">✅ License plates detected with bounding boxes and OCR text</p>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
