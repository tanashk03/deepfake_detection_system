import { useState, useRef, useEffect } from 'react';
import { analyzeVideo } from '../services/api';
import './LiveCamera.css';

function LiveCamera({ onStart, onComplete, onError, isAnalyzing }) {
    const [stream, setStream] = useState(null);
    const [isRecording, setIsRecording] = useState(false);
    const [recordedBlob, setRecordedBlob] = useState(null);
    const [countdown, setCountdown] = useState(null);
    const [recordingUrl, setRecordingUrl] = useState(null);

    const videoRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);

    useEffect(() => {
        return () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            if (recordingUrl) {
                URL.revokeObjectURL(recordingUrl);
            }
        };
    }, [stream, recordingUrl]);

    const startCamera = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: 640, height: 480 },
                audio: true
            });

            setStream(mediaStream);
            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
                videoRef.current.muted = true; // Mute live feed to prevent feedback
            }
        } catch (err) {
            console.error(err);
            onError({ message: 'Could not access camera. Please check permissions.' });
        }
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
        setRecordedBlob(null);
        if (recordingUrl) {
            URL.revokeObjectURL(recordingUrl);
            setRecordingUrl(null);
        }
    };

    const getSupportedMimeType = () => {
        const types = [
            'video/webm;codecs=vp8,opus',
            'video/webm',
            'video/mp4'
        ];
        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        return ''; // Let browser choose default
    };

    const startRecording = () => {
        if (!stream) return;

        chunksRef.current = [];
        const mimeType = getSupportedMimeType();
        const options = mimeType ? { mimeType } : {};

        try {
            const mediaRecorder = new MediaRecorder(stream, options);

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'video/webm' });
                setRecordedBlob(blob);
                const url = URL.createObjectURL(blob);
                setRecordingUrl(url);
                setIsRecording(false);
            };

            mediaRecorderRef.current = mediaRecorder;
            mediaRecorder.start(100);
            setIsRecording(true);

            // Record for 8 seconds (better for rPPG)
            let seconds = 8;
            setCountdown(seconds);

            const timer = setInterval(() => {
                seconds -= 1;
                setCountdown(seconds);

                if (seconds <= 0) {
                    clearInterval(timer);
                    stopRecording();
                }
            }, 1000);
        } catch (err) {
            console.error(err);
            onError({ message: 'Failed to start recording. MediaRecorder error.' });
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        setCountdown(null);
    };

    const handleAnalyze = async () => {
        if (!recordedBlob) return;

        onStart();

        try {
            const file = new File([recordedBlob], 'live_recording.webm', { type: 'video/webm' });
            // Use detailed mode for live camera as well
            const result = await analyzeVideo(file, { detailed: true });
            onComplete(result);
        } catch (err) {
            onError(err);
        }
    };

    const handleRetake = () => {
        setRecordedBlob(null);
        if (recordingUrl) {
            URL.revokeObjectURL(recordingUrl);
            setRecordingUrl(null);
        }
        // Force re-attach stream to video element
        if (videoRef.current && stream) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
        }
    };

    return (
        <div className="live-camera card">
            <h3>Live Camera Test</h3>
            <p>Record a short video clip (8s) for real-time analysis</p>

            <div className="camera-container">
                {!stream ? (
                    <div className="camera-placeholder" onClick={startCamera}>
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                            <circle cx="12" cy="13" r="4" />
                        </svg>
                        <span>Click to enable camera</span>
                    </div>
                ) : (
                    <div className="camera-view">
                        {recordedBlob && recordingUrl ? (
                            <video
                                src={recordingUrl}
                                className="camera-video recorded"
                                controls
                                playsInline
                            />
                        ) : (
                            <video
                                ref={videoRef}
                                autoPlay
                                muted
                                playsInline
                                className="camera-video live"
                            />
                        )}

                        {isRecording && countdown !== null && (
                            <div className="recording-indicator">
                                <span className="rec-dot"></span>
                                <span>Recording... {countdown}s</span>
                            </div>
                        )}
                    </div>
                )}
            </div>

            <div className="camera-actions">
                {!stream ? (
                    <button className="btn-primary" onClick={startCamera}>
                        Enable Camera
                    </button>
                ) : !recordedBlob ? (
                    <>
                        <button className="btn-secondary" onClick={stopCamera}>
                            Close Camera
                        </button>
                        {!isRecording ? (
                            <button className="btn-primary record-btn" onClick={startRecording}>
                                <span className="record-icon"></span>
                                Record 8s Clip
                            </button>
                        ) : (
                            <button className="btn-primary" onClick={stopRecording}>
                                Stop Recording
                            </button>
                        )}
                    </>
                ) : (
                    <>
                        <button className="btn-secondary" onClick={handleRetake} disabled={isAnalyzing}>
                            Retake
                        </button>
                        <button className="btn-primary" onClick={handleAnalyze} disabled={isAnalyzing}>
                            {isAnalyzing ? (
                                <>
                                    <span className="spinner"></span>
                                    Analyzing...
                                </>
                            ) : (
                                'Analyze Recording'
                            )}
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}

export default LiveCamera;
