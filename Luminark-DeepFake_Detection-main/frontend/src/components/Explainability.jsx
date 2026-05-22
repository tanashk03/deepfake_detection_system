import './Explainability.css';

function Explainability({ result }) {
    if (!result || !result.modality_contributions) {
        return null;
    }

    const contributions = result.modality_contributions;
    const hasContributions = Object.keys(contributions).length > 0;

    const modalityLabels = {
        video: 'Visual Analysis',
        audio: 'Audio Analysis',
        rppg: 'Physiological Signals',
        lipsync: 'Lip-Sync Check'
    };

    const modalityIcons = {
        video: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                <line x1="8" y1="21" x2="16" y2="21" />
                <line x1="12" y1="17" x2="12" y2="21" />
            </svg>
        ),
        audio: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
            </svg>
        ),
        rppg: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
        ),
        lipsync: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z" />
                <path d="M8 14s1.5 2 4 2 4-2 4-2" />
                <line x1="9" y1="9" x2="9.01" y2="9" />
                <line x1="15" y1="9" x2="15.01" y2="9" />
            </svg>
        )
    };

    return (
        <div className="explainability card">
            <h4>Analysis Breakdown</h4>

            {/* Heatmap Placeholder */}
            <div className="heatmap-section">
                <div className="heatmap-placeholder">
                    <div className="heatmap-grid">
                        {[...Array(16)].map((_, i) => (
                            <div
                                key={i}
                                className="heatmap-cell"
                                style={{
                                    opacity: 0.2 + Math.random() * 0.6,
                                    backgroundColor: result.verdict === 'FAKE'
                                        ? 'var(--color-fake)'
                                        : result.verdict === 'REAL'
                                            ? 'var(--color-real)'
                                            : 'var(--color-uncertain)'
                                }}
                            />
                        ))}
                    </div>
                    <span className="heatmap-label">Region attention map</span>
                </div>
            </div>

            {/* Modality Breakdown */}
            {hasContributions && (
                <div className="modality-breakdown">
                    <h5>Detection Channels</h5>
                    <div className="modality-list">
                        {Object.entries(contributions)
                            .sort((a, b) => b[1] - a[1])
                            .map(([modality, value]) => (
                                <div key={modality} className="modality-item">
                                    <div className="modality-icon">
                                        {modalityIcons[modality] || modalityIcons.video}
                                    </div>
                                    <div className="modality-info">
                                        <span className="modality-name">
                                            {modalityLabels[modality] || modality}
                                        </span>
                                        <div className="modality-bar">
                                            <div
                                                className="modality-bar-fill"
                                                style={{ width: `${Math.round(value * 100)}%` }}
                                            />
                                        </div>
                                    </div>
                                    <span className="modality-value">
                                        {Math.round(value * 100)}%
                                    </span>
                                </div>
                            ))}
                    </div>
                </div>
            )}

            {/* Uncertainty */}
            {result.uncertainty !== undefined && (
                <div className="uncertainty-section">
                    <div className="uncertainty-header">
                        <span>Model Uncertainty</span>
                        <span className={`uncertainty-level ${result.uncertainty < 0.2 ? 'low' :
                                result.uncertainty < 0.4 ? 'medium' : 'high'
                            }`}>
                            {result.uncertainty < 0.2 ? 'Low' :
                                result.uncertainty < 0.4 ? 'Medium' : 'High'}
                        </span>
                    </div>
                    <div className="uncertainty-bar">
                        <div
                            className="uncertainty-bar-fill"
                            style={{ width: `${Math.round(result.uncertainty * 100)}%` }}
                        />
                    </div>
                    <p className="uncertainty-note">
                        {result.uncertainty < 0.2
                            ? 'High confidence in this assessment.'
                            : result.uncertainty < 0.4
                                ? 'Moderate confidence. Some ambiguity detected.'
                                : 'Low confidence. Consider additional verification.'}
                    </p>
                </div>
            )}
        </div>
    );
}

export default Explainability;
