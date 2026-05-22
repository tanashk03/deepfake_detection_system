import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Upload, Video, Camera, X, Play, Loader,
    CheckCircle, AlertTriangle, HelpCircle,
    BarChart3, Shield, RefreshCw
} from 'lucide-react';
import './Analyze.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || 'lum_test_key_12345';

export default function Analyze() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [isRecording, setIsRecording] = useState(false);
    const [isCameraOpen, setIsCameraOpen] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // Analysis State
    const [jobId, setJobId] = useState(null);
    const [analysisStep, setAnalysisStep] = useState('Queued');
    const [progress, setProgress] = useState(0);
    const [currentStepIndex, setCurrentStepIndex] = useState(0);

    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);

    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);
    const timerRef = useRef(null);
    const pollIntervalRef = useRef(null);

    // Mapped steps for the UI Stepper
    const UI_STEPS = [
        { label: 'Upload', id: 'queued' },
        { label: 'Extract', id: 'extracting' },
        { label: 'Analyze', id: 'running' },
        { label: 'Fuse', id: 'fusing' },
        { label: 'Result', id: 'finalizing' }
    ];

    // Map backend steps to UI indices
    const getStepIndex = (step) => {
        const s = step?.toLowerCase() || '';
        if (s.includes('extract')) return 1;
        if (s.includes('standard') || s.includes('advanced')) return 2;
        if (s.includes('fusing')) return 3;
        if (s.includes('finaliz')) return 4;
        return 0; // Default/Start
    };

    // Cleanup polling on unmount
    useEffect(() => {
        return () => {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        };
    }, []);

    // Handle file selection
    const handleFileSelect = useCallback((selectedFile) => {
        if (selectedFile && selectedFile.type.startsWith('video/')) {
            setFile(selectedFile);
            setPreview(URL.createObjectURL(selectedFile));
            setResult(null);
            setError(null);
        }
    }, []);

    // Drag and drop handlers
    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files?.[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    };

    // Camera handling
    const openCamera = async () => {
        try {
            setIsCameraOpen(true);
            setTimeout(async () => {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: true
                });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.play();
                }
            }, 100);
        } catch (err) {
            setError('Camera access denied. Please allow camera permissions.');
            setIsCameraOpen(false);
        }
    };

    const startRecording = () => {
        if (!videoRef.current?.srcObject) return;

        const stream = videoRef.current.srcObject;
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        chunksRef.current = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                chunksRef.current.push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(chunksRef.current, { type: 'video/webm' });
            const file = new File([blob], 'recording.webm', { type: 'video/webm' });
            handleFileSelect(file);
            closeCamera();
        };

        mediaRecorder.start();
        setIsRecording(true);
        setRecordingTime(0);
        timerRef.current = setInterval(() => {
            setRecordingTime(prev => prev + 1);
        }, 1000);
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            clearInterval(timerRef.current);
        }
    };

    const closeCamera = () => {
        if (videoRef.current?.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach(track => track.stop());
        }
        setIsCameraOpen(false);
        setIsRecording(false);
        clearInterval(timerRef.current);
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // ------------------------------------------------------------------------
    // Real Analysis Logic (Polling)
    // ------------------------------------------------------------------------

    const analyzeVideo = async () => {
        if (!file) return;

        setIsAnalyzing(true);
        setError(null);
        setProgress(0);
        setAnalysisStep('Initiating...');
        setCurrentStepIndex(0);

        const formData = new FormData();
        formData.append('video', file);

        try {
            // 1. Start Job
            const startRes = await fetch(`${API_URL}/analyze/start`, {
                method: 'POST',
                headers: { 'X-API-Key': API_KEY },
                body: formData,
            });

            if (!startRes.ok) throw new Error('Failed to start analysis');

            const { job_id } = await startRes.json();
            setJobId(job_id);

            // 2. Poll Status
            pollIntervalRef.current = setInterval(async () => {
                try {
                    const statusRes = await fetch(`${API_URL}/analyze/status/${job_id}`, {
                        headers: { 'X-API-Key': API_KEY }
                    });
                    if (!statusRes.ok) return;

                    const statusData = await statusRes.json();

                    if (statusData.status === 'failed') {
                        throw new Error(statusData.error || 'Analysis failed');
                    }

                    // Update UI
                    setAnalysisStep(statusData.step || 'Processing...');
                    setProgress(statusData.progress);
                    setCurrentStepIndex(getStepIndex(statusData.step));

                    // Check completion
                    if (statusData.status === 'completed') {
                        clearInterval(pollIntervalRef.current);
                        fetchResult(job_id);
                    }

                } catch (pollErr) {
                    console.error("Polling error:", pollErr);
                    clearInterval(pollIntervalRef.current);
                    setError(pollErr.message);
                    setIsAnalyzing(false);
                }
            }, 1000); // Poll every 1s

        } catch (err) {
            console.error(err);
            setError('Failed to start analysis. Please try again.');
            setIsAnalyzing(false);
        }
    };

    const fetchResult = async (id) => {
        try {
            const res = await fetch(`${API_URL}/analyze/result/${id}`, {
                headers: { 'X-API-Key': API_KEY }
            });
            if (!res.ok) throw new Error('Failed to fetch results');
            const data = await res.json();

            // Brief delay to show 100%
            setTimeout(() => {
                setResult(data);
                setIsAnalyzing(false);
            }, 500);
        } catch (err) {
            setError(err.message);
            setIsAnalyzing(false);
        }
    };

    const reset = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };

    const getVerdictStyle = (verdict) => {
        switch (verdict?.toUpperCase()) {
            case 'FAKE': return { color: 'var(--danger)', icon: AlertTriangle };
            case 'REAL': return { color: 'var(--success)', icon: CheckCircle };
            default: return { color: 'var(--text-secondary)', icon: HelpCircle };
        }
    };

    return (
        <div className="analyze-page">
            <div className="container">
                <motion.div
                    className="page-header"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <h1>Analyze Video</h1>
                    <p>Upload or record a video to detect deepfake manipulation</p>
                </motion.div>

                <div className="analyze-content">
                    {/* Left Column: Input */}
                    <motion.div
                        className="upload-section"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.1 }}
                    >
                        {!preview ? (
                            <>
                                {/* Drag & Drop Zone */}
                                <div
                                    className={`upload-zone ${dragActive ? 'active' : ''}`}
                                    onDragEnter={handleDrag}
                                    onDragLeave={handleDrag}
                                    onDragOver={handleDrag}
                                    onDrop={handleDrop}
                                    onClick={() => fileInputRef.current?.click()}
                                >
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept="video/*"
                                        onChange={(e) => handleFileSelect(e.target.files?.[0])}
                                        hidden
                                    />
                                    <Upload className="upload-icon" size={48} />
                                    <h3>Drop your video here</h3>
                                    <p>or click to browse</p>
                                    <span className="upload-formats">MP4, AVI, MOV, WebM</span>
                                </div>

                                <div className="upload-divider">
                                    <span>or record directly</span>
                                </div>

                                {/* Record Option */}
                                <div className="record-section">
                                    {!isCameraOpen ? (
                                        <button className="btn btn-secondary record-btn" onClick={openCamera}>
                                            <Camera size={20} />
                                            Open Camera
                                        </button>
                                    ) : (
                                        <div className="recording-ui">
                                            <video ref={videoRef} className="recording-preview" muted playsInline />
                                            <div className="recording-overlays">
                                                {isRecording && (
                                                    <div className="recording-timer">
                                                        <div className="recording-dot" />
                                                        {formatTime(recordingTime)}
                                                    </div>
                                                )}
                                            </div>
                                            <div className="recording-controls">
                                                {!isRecording ? (
                                                    <>
                                                        <button className="btn btn-primary start-rec-btn" onClick={startRecording}>
                                                            <div className="rec-icon" />
                                                            Start Recording
                                                        </button>
                                                        <button className="btn btn-ghost cancel-btn" onClick={closeCamera}>
                                                            Cancel
                                                        </button>
                                                    </>
                                                ) : (
                                                    <button className="btn btn-danger stop-btn" onClick={stopRecording}>
                                                        <div className="stop-icon" />
                                                        Stop Recording
                                                    </button>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </>
                        ) : (
                            /* Preview Section */
                            <div className="preview-section">
                                {!isAnalyzing ? (
                                    <>
                                        <video src={preview} className="video-preview" controls />
                                        <div className="preview-info">
                                            <Video size={18} />
                                            <span>{file?.name}</span>
                                            <button className="btn-icon" onClick={reset}>
                                                <X size={18} />
                                            </button>
                                        </div>

                                        {!result && (
                                            <button className="btn btn-primary analyze-btn" onClick={analyzeVideo}>
                                                <Play size={20} />
                                                Start Analysis
                                            </button>
                                        )}
                                    </>
                                ) : (
                                    /* NEW HORIZONTAL STEPPER */
                                    <div className="analysis-progress-container">
                                        <div className="progress-header">
                                            <h3>{analysisStep}</h3>
                                            <span className="progress-percentage">{Math.round(progress)}%</span>
                                        </div>

                                        {/* Horizontal Stepper UI */}
                                        <div className="horizontal-stepper">
                                            {/* Progress Track Line */}
                                            <div className="stepper-track">
                                                <motion.div
                                                    className="stepper-fill"
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${progress}%` }}
                                                    transition={{ duration: 0.5 }}
                                                />
                                            </div>

                                            {/* Step Nodes */}
                                            <div className="stepper-nodes">
                                                {UI_STEPS.map((step, index) => {
                                                    const isCompleted = index < currentStepIndex;
                                                    const isActive = index === currentStepIndex;

                                                    return (
                                                        <div
                                                            key={index}
                                                            className={`step-node ${isCompleted ? 'completed' : ''} ${isActive ? 'active' : ''}`}
                                                        >
                                                            <div className="node-circle">
                                                                {isCompleted ? <CheckCircle size={14} /> : (index + 1)}
                                                            </div>
                                                            <span className="step-label">{step.label}</span>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>

                                        <div className="loading-spinner-container">
                                            <Loader className="spin" size={24} />
                                            <p className="sub-text">Please keep this window open</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </motion.div>

                    {/* Right Column: Results */}
                    <div className="results-column">
                        <AnimatePresence mode="wait">
                            {result ? (
                                <motion.div
                                    key="results"
                                    className="results-section"
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -20 }}
                                >
                                    {/* Verdict Card */}
                                    <div className="verdict-card" style={{ '--verdict-color': getVerdictStyle(result.verdict).color }}>
                                        <div className="verdict-icon">
                                            {(() => {
                                                const Icon = getVerdictStyle(result.verdict).icon;
                                                return <Icon size={48} />;
                                            })()}
                                        </div>
                                        <div className="verdict-text">
                                            <span className="verdict-label">Verdict</span>
                                            <h2 className="verdict-value">{result.verdict}</h2>
                                        </div>
                                        <div className="confidence-meter">
                                            <div className="confidence-bar">
                                                <motion.div
                                                    className="confidence-fill"
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${result.confidence}%` }}
                                                    transition={{ duration: 1, ease: 'easeOut' }}
                                                />
                                            </div>
                                            <span className="confidence-value">{result.confidence}% Confidence</span>
                                        </div>
                                    </div>

                                    {/* Explanation */}
                                    {result.explanation && (
                                        <div className="explanation-card">
                                            <Shield size={20} />
                                            <p>{result.explanation}</p>
                                        </div>
                                    )}

                                    {/* Model Contributions */}
                                    {result.modality_contributions && (
                                        <div className="contributions-card">
                                            <div className="card-header">
                                                <BarChart3 size={20} />
                                                <h3>Model Contributions</h3>
                                            </div>
                                            <div className="contributions-list">
                                                {Object.entries(result.modality_contributions)
                                                    .sort(([, a], [, b]) => b - a)
                                                    .map(([model, contribution]) => (
                                                        <div key={model} className="contribution-item">
                                                            <span className="contribution-name">{model}</span>
                                                            <div className="contribution-bar">
                                                                <motion.div
                                                                    className="contribution-fill"
                                                                    initial={{ width: 0 }}
                                                                    animate={{ width: `${contribution * 100}%` }}
                                                                    transition={{ duration: 0.8 }}
                                                                />
                                                            </div>
                                                            <span className="contribution-value">
                                                                {(contribution * 100).toFixed(0)}%
                                                            </span>
                                                        </div>
                                                    ))}
                                            </div>
                                        </div>
                                    )}

                                    <button className="btn btn-secondary reset-btn" onClick={reset}>
                                        <RefreshCw size={18} />
                                        Analyze Another Video
                                    </button>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="info"
                                    className="info-panel"
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                >
                                    <h2>System Capabilities</h2>
                                    <div className="info-grid">
                                        <div className="info-item">
                                            <Video className="info-icon" size={24} />
                                            <div>
                                                <h3>Multi-Format Support</h3>
                                                <p>Analyze MP4, WebM, and MOV files up to 50MB.</p>
                                            </div>
                                        </div>
                                        <div className="info-item">
                                            <BarChart3 className="info-icon" size={24} />
                                            <div>
                                                <h3>6-Model Ensemble</h3>
                                                <p>Cross-references audio, visual, and biological signals.</p>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="info-note">
                                        <AlertTriangle size={16} />
                                        <p>For best results, ensure the face is clearly visible and lighting is adequate.</p>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>

                {/* Sponsor Banner - Always Visible */}
                <motion.div
                    className="sponsor-banner sponsor-banner-page"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <div className="sponsor-content">
                        <p className="sponsor-text">
                            ⚡ Running on free-tier infrastructure. Help us upgrade to faster GPU servers!
                        </p>
                        <a
                            href="https://github.com/sponsors/IsVohi"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="sponsor-btn"
                        >
                            ❤️ Sponsor This Project
                        </a>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
