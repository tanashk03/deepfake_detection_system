import { useState, useRef } from 'react';
import { analyzeVideo } from '../services/api';
import './VideoUpload.css';

function VideoUpload({ onStart, onComplete, onError, isAnalyzing }) {
    const [dragActive, setDragActive] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const fileInputRef = useRef(null);

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

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = (file) => {
        // Validate file type
        const validTypes = ['video/mp4', 'video/quicktime', 'video/webm', 'video/x-msvideo'];
        if (!validTypes.includes(file.type)) {
            onError({ message: 'Please upload a valid video file (MP4, MOV, WebM, AVI)' });
            return;
        }

        // Validate file size (500 MB)
        if (file.size > 500 * 1024 * 1024) {
            onError({ message: 'File too large. Maximum size is 500 MB.' });
            return;
        }

        setSelectedFile(file);

        // Create preview
        const url = URL.createObjectURL(file);
        setPreview(url);
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;

        onStart();

        try {
            // Use detailed mode to get calibrated confidence, XAI, and escalation info
            const result = await analyzeVideo(selectedFile, { detailed: true });
            onComplete(result);
        } catch (err) {
            onError(err);
        }
    };

    const handleClear = () => {
        setSelectedFile(null);
        setPreview(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className="video-upload card">
            <h3>Upload Video</h3>
            <p>Drag and drop or click to select a video file for analysis</p>

            {!selectedFile ? (
                <div
                    className={`drop-zone ${dragActive ? 'active' : ''}`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                    <span className="drop-text">Drop video here or click to browse</span>
                    <span className="drop-hint">MP4, MOV, WebM, AVI up to 500 MB</span>
                </div>
            ) : (
                <div className="preview-container">
                    <video
                        src={preview}
                        className="video-preview"
                        controls
                        muted
                    />
                    <div className="file-info">
                        <span className="file-name">{selectedFile.name}</span>
                        <span className="file-size">
                            {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                        </span>
                    </div>
                </div>
            )}

            <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/quicktime,video/webm,video/x-msvideo"
                onChange={handleChange}
                style={{ display: 'none' }}
            />

            <div className="upload-actions">
                {selectedFile && (
                    <>
                        <button
                            className="btn-secondary"
                            onClick={handleClear}
                            disabled={isAnalyzing}
                        >
                            Clear
                        </button>
                        <button
                            className="btn-primary"
                            onClick={handleAnalyze}
                            disabled={isAnalyzing}
                        >
                            {isAnalyzing ? (
                                <>
                                    <span className="spinner"></span>
                                    Analyzing...
                                </>
                            ) : (
                                'Analyze Video'
                            )}
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}

export default VideoUpload;
