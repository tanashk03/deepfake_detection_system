import { motion } from 'framer-motion';
import {
    Cpu, Layers, Zap, Database, Server,
    ArrowRight, Code, Terminal, Shield
} from 'lucide-react';
import './Docs.css';

const models = [
    {
        name: 'VideoMAE',
        weight: '25%',
        architecture: 'Vision Transformer',
        description: 'State-of-the-art video understanding model from HuggingFace that analyzes spatiotemporal patterns across frames.',
    },
    {
        name: 'Spatial',
        weight: '25%',
        architecture: 'EfficientNet-B0',
        description: 'Detects frame-level artifacts, blending inconsistencies, and texture anomalies in individual frames.',
    },
    {
        name: 'Frequency',
        weight: '25%',
        architecture: 'EfficientNet + FFT',
        description: 'Analyzes frequency domain using FFT/DCT to reveal GAN fingerprints and spectral artifacts.',
    },
    {
        name: 'Physiological',
        weight: '10%',
        architecture: 'CNN + BiLSTM',
        description: 'Monitors biological patterns like pulse variations and skin color changes across video frames.',
    },
    {
        name: 'Temporal',
        weight: '7%',
        architecture: 'Conv3D Network',
        description: 'Identifies motion inconsistencies, flickering, and unnatural frame transitions.',
    },
    {
        name: 'Audio',
        weight: '1-25%',
        architecture: 'WavLM Transformer',
        description: 'Detects voice synthesis artifacts, unnatural prosody, and audio-visual sync issues.',
    },
];

const apiEndpoints = [
    {
        method: 'POST',
        endpoint: '/explain',
        description: 'Analyze video and return detailed verdict with model contributions',
    },
    {
        method: 'GET',
        endpoint: '/health',
        description: 'Health check endpoint for monitoring',
    },
    {
        method: 'GET',
        endpoint: '/metrics',
        description: 'Prometheus metrics for observability',
    },
];

export default function Docs() {
    return (
        <div className="docs-page">
            <div className="container">
                <motion.div
                    className="page-header"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <h1>Documentation</h1>
                    <p>Technical architecture and API reference</p>
                </motion.div>

                {/* Architecture Overview */}
                <motion.section
                    className="docs-section"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    <div className="section-title">
                        <Cpu size={24} />
                        <h2>System Architecture</h2>
                    </div>

                    <div className="architecture-diagram">
                        <div className="diagram-flow">
                            <div className="diagram-node input">
                                <Database size={20} />
                                <span>Video Input</span>
                            </div>
                            <ArrowRight className="diagram-arrow" />
                            <div className="diagram-node process">
                                <Layers size={20} />
                                <span>Preprocessing</span>
                            </div>
                            <ArrowRight className="diagram-arrow" />
                            <div className="diagram-node models">
                                <Zap size={20} />
                                <span>6 AI Models</span>
                            </div>
                            <ArrowRight className="diagram-arrow" />
                            <div className="diagram-node fusion">
                                <Server size={20} />
                                <span>Late Fusion</span>
                            </div>
                            <ArrowRight className="diagram-arrow" />
                            <div className="diagram-node output">
                                <Shield size={20} />
                                <span>Verdict</span>
                            </div>
                        </div>
                    </div>

                    <div className="architecture-details">
                        <div className="detail-card">
                            <h4>Preprocessing</h4>
                            <p>Frame extraction at 5 FPS, resize to 224×224, ImageNet normalization</p>
                        </div>
                        <div className="detail-card">
                            <h4>Parallel Inference</h4>
                            <p>All 6 models analyze video simultaneously for maximum efficiency</p>
                        </div>
                        <div className="detail-card">
                            <h4>Late Fusion</h4>
                            <p>Weighted score aggregation with uncertainty-aware calibration</p>
                        </div>
                    </div>
                </motion.section>

                {/* Models Section */}
                <motion.section
                    className="docs-section"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    <div className="section-title">
                        <Layers size={24} />
                        <h2>Detection Models</h2>
                    </div>

                    <div className="models-grid">
                        {models.map((model, index) => (
                            <motion.div
                                key={model.name}
                                className="model-card"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.1 * index }}
                            >
                                <div className="model-header">
                                    <h3>{model.name}</h3>
                                    <span className="model-weight">{model.weight}</span>
                                </div>
                                <span className="model-arch">{model.architecture}</span>
                                <p>{model.description}</p>
                            </motion.div>
                        ))}
                    </div>
                </motion.section>

                {/* API Reference */}
                <motion.section
                    className="docs-section"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <div className="section-title">
                        <Code size={24} />
                        <h2>API Reference</h2>
                    </div>

                    <div className="api-endpoints">
                        {apiEndpoints.map((api) => (
                            <div key={api.endpoint} className="api-card">
                                <div className="api-header">
                                    <span className={`api-method ${api.method.toLowerCase()}`}>
                                        {api.method}
                                    </span>
                                    <code className="api-endpoint">{api.endpoint}</code>
                                </div>
                                <p>{api.description}</p>
                            </div>
                        ))}
                    </div>

                    {/* Example Request */}
                    <div className="code-block">
                        <div className="code-header">
                            <Terminal size={16} />
                            <span>Example Request</span>
                        </div>
                        <pre>
                            {`curl -X POST http://localhost:8000/explain \\
  -H "X-API-Key: your_api_key" \\
  -F "video=@your_video.mp4"`}
                        </pre>
                    </div>

                    {/* Example Response */}
                    <div className="code-block">
                        <div className="code-header">
                            <Code size={16} />
                            <span>Example Response</span>
                        </div>
                        <pre>
                            {`{
  "verdict": "FAKE",
  "confidence": 95,
  "score": 0.42,
  "modality_contributions": {
    "videomae": 0.25,
    "spatial": 0.25,
    "frequency": 0.25,
    "temporal": 0.10,
    "physiological": 0.10
  },
  "explanation": "Spatial analysis detected blending artifacts."
}`}
                        </pre>
                    </div>
                </motion.section>
            </div>
        </div>
    );
}
