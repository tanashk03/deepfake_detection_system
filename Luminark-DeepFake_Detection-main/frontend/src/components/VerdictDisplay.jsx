import './VerdictDisplay.css';

function VerdictDisplay({ result, isAnalyzing }) {
    if (isAnalyzing) {
        return (
            <div className="verdict-display card analyzing">
                <div className="analyzing-content">
                    <div className="analyzing-spinner">
                        <div className="spinner-ring"></div>
                        <div className="spinner-ring"></div>
                        <div className="spinner-ring"></div>
                    </div>
                    <h3>Analyzing Video</h3>
                    <p>Running multimodal detection across video, audio, and physiological signals...</p>
                </div>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="verdict-display card empty">
                <div className="empty-content">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                        <line x1="12" y1="17" x2="12.01" y2="17" />
                    </svg>
                    <h3>Ready to Analyze</h3>
                    <p>Upload a video or use the live camera to detect deepfakes</p>
                </div>
            </div>
        );
    }

    const verdictClass = result.verdict.toLowerCase();
    const verdictLabels = {
        REAL: 'Authentic',
        FAKE: 'Manipulated',
        INCONCLUSIVE: 'Uncertain'
    };

    return (
        <div className={`verdict-display card ${verdictClass}`}>
            <div className="verdict-header">
                <div className={`traffic-light ${verdictClass}`}>
                    <div className="light red"></div>
                    <div className="light yellow"></div>
                    <div className="light green"></div>
                </div>

                <div className="verdict-text">
                    <span className="verdict-label">Verdict</span>
                    <h2 className="verdict-value">{verdictLabels[result.verdict] || result.verdict}</h2>
                </div>

                <div className="confidence-badge">
                    <span className="confidence-value">{result.calibrated_confidence || result.confidence}%</span>
                    <span className="confidence-label">
                        {result.calibrated_confidence ? 'calibrated' : 'confidence'}
                    </span>
                </div>
            </div>

            {/* Modality Breakdown */}
            {result.modality_contributions && (
                <div className="modality-breakdown">
                    <h4>Model Analysis Breakdown</h4>
                    <div className="modality-bars">
                        {Object.entries(result.modality_contributions).map(([modality, score]) => (
                            <div key={modality} className="modality-row">
                                <span className="modality-name">{modality.charAt(0).toUpperCase() + modality.slice(1)}</span>
                                <div className="modality-bar-container">
                                    <div
                                        className="modality-bar"
                                        style={{ width: `${Math.min(100, Math.max(5, score * 100))}%` }}
                                    ></div>
                                </div>
                                <span className="modality-score">{(score * 100).toFixed(0)}% Impact</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <p className="verdict-explanation">{result.explanation}</p>

            {/* XAI Summary */}
            {result.xai_summary && (
                <div className="xai-summary">
                    <span className="xai-label">Analysis Rationale</span>
                    <p>{result.xai_summary}</p>
                </div>
            )}

            {/* Cloud Escalation Indicator */}
            {result.should_escalate && (
                <div className="escalation-notice">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M7.5 21L3 16.5L7.5 12M9 16.5H21M16.5 3L21 7.5L16.5 12M15 7.5H3" />
                    </svg>
                    <span>Recommended for cloud verification</span>
                </div>
            )}

            {result.processing_time_ms && (
                <div className="processing-time">
                    Analyzed in {(result.processing_time_ms / 1000).toFixed(1)}s
                </div>
            )}
        </div>
    );
}

export default VerdictDisplay;
