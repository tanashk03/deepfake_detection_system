import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
    Shield, Zap, Eye, Brain, Waves, Activity,
    CheckCircle, ArrowRight, Play
} from 'lucide-react';
import './Landing.css';

const features = [
    {
        icon: Eye,
        title: 'VideoMAE Transformer',
        description: 'State-of-the-art vision transformer for spatiotemporal pattern detection',
    },
    {
        icon: Brain,
        title: 'Spatial Analysis',
        description: 'EfficientNet-B0 detects frame-level artifacts and blending inconsistencies',
    },
    {
        icon: Waves,
        title: 'Frequency Detection',
        description: 'FFT/DCT spectral analysis reveals GAN fingerprints and anomalies',
    },
    {
        icon: Activity,
        title: 'Temporal Patterns',
        description: 'Conv3D network identifies motion inconsistencies across frames',
    },
    {
        icon: Zap,
        title: 'Physiological Signals',
        description: 'CNN-BiLSTM analyzes biological patterns for authenticity verification',
    },
    {
        icon: Shield,
        title: 'Audio Analysis',
        description: 'WavLM transformer detects voice synthesis and manipulation',
    },
];

const stats = [
    { value: '95%', label: 'Accuracy' },
    { value: '6', label: 'AI Models' },
    { value: '<5s', label: 'Analysis Time' },
    { value: '100%', label: 'Privacy Safe' },
];

export default function Landing() {
    return (
        <div className="landing">
            {/* Hero Section */}
            <section className="hero">
                <div className="hero-bg">
                    <div className="hero-gradient" />
                    <div className="hero-grid" />
                </div>

                <div className="hero-content">
                    <motion.div
                        className="hero-badge"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <Shield size={16} />
                        <span>AI-Powered Detection</span>
                    </motion.div>

                    <motion.h1
                        className="hero-title"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.1 }}
                    >
                        Detect Deepfakes with
                        <span className="gradient-text"> Confidence</span>
                    </motion.h1>

                    <motion.p
                        className="hero-subtitle"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                    >
                        Luminark combines 6 specialized AI models to identify manipulated videos
                        with industry-leading 95% accuracy. Protect yourself from synthetic media.
                    </motion.p>

                    <motion.div
                        className="hero-actions"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.3 }}
                    >
                        <Link to="/analyze" className="btn btn-primary btn-lg">
                            <Play size={20} />
                            Start Analysis
                        </Link>
                        <Link to="/docs" className="btn btn-secondary btn-lg">
                            Learn More
                            <ArrowRight size={18} />
                        </Link>
                    </motion.div>
                </div>
            </section>

            {/* Stats Section */}
            <section className="stats">
                <div className="container">
                    <div className="stats-grid">
                        {stats.map((stat, index) => (
                            <motion.div
                                key={stat.label}
                                className="stat-card"
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.5, delay: index * 0.1 }}
                            >
                                <span className="stat-value">{stat.value}</span>
                                <span className="stat-label">{stat.label}</span>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="features">
                <div className="container">
                    <motion.div
                        className="section-header"
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                    >
                        <h2>Multimodal Detection Engine</h2>
                        <p>Six specialized AI models work together for comprehensive analysis</p>
                    </motion.div>

                    <div className="features-grid">
                        {features.map((feature, index) => (
                            <motion.div
                                key={feature.title}
                                className="feature-card"
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.5, delay: index * 0.1 }}
                                whileHover={{ y: -5 }}
                            >
                                <div className="feature-icon">
                                    <feature.icon size={24} />
                                </div>
                                <h3>{feature.title}</h3>
                                <p>{feature.description}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* How It Works */}
            <section className="how-it-works">
                <div className="container">
                    <motion.div
                        className="section-header"
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                    >
                        <h2>How It Works</h2>
                        <p>Advanced deepfake detection in three simple steps</p>
                    </motion.div>

                    <div className="steps">
                        {[
                            {
                                num: '01',
                                icon: Play,
                                title: 'Upload or Record Video',
                                desc: 'Simply drag and drop your video file (MP4, WebM, MOV) or use the live camera recording feature. Maximum file size: 500MB.',
                                detail: 'Supported formats: MP4, AVI, MOV, WebM • Privacy guaranteed: Videos deleted immediately after analysis'
                            },
                            {
                                num: '02',
                                icon: Brain,
                                title: 'Multi-Model AI Analysis',
                                desc: 'Our ensemble of 6 AI models simultaneously analyzes spatial patterns, temporal inconsistencies, frequency anomalies, physiological signals, and audio artifacts.',
                                detail: 'VideoMAE • EfficientNet • CNN-LSTM • FFT/DCT Analysis • WavLM • Lip-sync Detection'
                            },
                            {
                                num: '03',
                                icon: CheckCircle,
                                title: 'Get Detailed Results',
                                desc: 'Receive a comprehensive verdict (Real/Fake) with confidence score, detailed explanation, and per-model contribution breakdown in under 5 seconds.',
                                detail: 'Real-time results • Model transparency • Explainable AI • Downloadable report'
                            },
                        ].map((step, index) => (
                            <motion.div
                                key={step.num}
                                className="step"
                                initial={{ opacity: 0, x: -20 }}
                                whileInView={{ opacity: 1, x: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.5, delay: index * 0.2 }}
                            >
                                <div className="step-icon-circle">
                                    <step.icon size={24} />
                                </div>
                                <div className="step-content">
                                    <span className="step-num">{step.num}</span>
                                    <h4>{step.title}</h4>
                                    <p>{step.desc}</p>
                                    <span className="step-detail">{step.detail}</span>
                                </div>
                                {index < 2 && <div className="step-connector" />}
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="cta">
                <div className="container">
                    <motion.div
                        className="cta-content"
                        initial={{ opacity: 0, scale: 0.95 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                    >
                        <CheckCircle className="cta-icon" size={48} />
                        <h2>Ready to Detect Deepfakes?</h2>
                        <p>Upload your video and get results in seconds</p>
                        <Link to="/analyze" className="btn btn-primary btn-lg">
                            <Play size={20} />
                            Analyze Video Now
                        </Link>
                    </motion.div>
                </div>
            </section>
        </div>
    );
}
