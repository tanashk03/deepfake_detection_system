import { motion } from 'framer-motion';
import { Shield, Trash2, Lock, Eye, Server, Mail } from 'lucide-react';
import './Privacy.css';

const policies = [
    {
        icon: Trash2,
        title: 'Immediate Deletion',
        description: 'All uploaded videos are permanently deleted immediately after analysis. We do not store, save, or retain any video content on our servers.',
    },
    {
        icon: Lock,
        title: 'Secure Processing',
        description: 'Your video is processed in an isolated environment using encrypted connections (HTTPS/TLS). No data is transmitted to third parties.',
    },
    {
        icon: Eye,
        title: 'No Tracking',
        description: 'We do not use cookies, analytics, or tracking mechanisms. Your analysis sessions are completely anonymous.',
    },
    {
        icon: Server,
        title: 'Local Processing',
        description: 'All AI analysis happens locally on our secure servers. Your video never leaves our controlled infrastructure during processing.',
    },
];

export default function Privacy() {
    return (
        <div className="privacy-page">
            <div className="container">
                <motion.div
                    className="page-header"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <Shield className="page-icon" size={48} />
                    <h1>Privacy Policy</h1>
                    <p>Your privacy is our priority. Here's how we handle your data.</p>
                </motion.div>

                {/* Key Policies */}
                <motion.div
                    className="policies-grid"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    {policies.map((policy, index) => (
                        <motion.div
                            key={policy.title}
                            className="policy-card"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 * (index + 1) }}
                        >
                            <div className="policy-icon">
                                <policy.icon size={24} />
                            </div>
                            <h3>{policy.title}</h3>
                            <p>{policy.description}</p>
                        </motion.div>
                    ))}
                </motion.div>

                {/* Detailed Policy */}
                <motion.div
                    className="policy-details"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <section className="policy-section">
                        <h2>Data We Collect</h2>
                        <p>
                            Luminark collects only the minimum data necessary to perform deepfake analysis:
                        </p>
                        <ul>
                            <li>Video files uploaded for analysis (deleted immediately after processing)</li>
                            <li>Basic request metadata (timestamp, file size) for rate limiting purposes</li>
                        </ul>
                        <p>
                            We do <strong>not</strong> collect:
                        </p>
                        <ul>
                            <li>Personal information or account data</li>
                            <li>IP addresses linked to analysis requests</li>
                            <li>Cookies or browser fingerprints</li>
                            <li>Any content from your videos</li>
                        </ul>
                    </section>

                    <section className="policy-section">
                        <h2>Data Retention</h2>
                        <p>
                            <strong>Videos:</strong> Deleted immediately upon completion of analysis. No copies are retained.
                        </p>
                        <p>
                            <strong>Analysis Results:</strong> Results are returned to you in real-time and are not stored on our servers.
                        </p>
                        <p>
                            <strong>Logs:</strong> Minimal server logs for debugging purposes are retained for 24 hours and contain no personally identifiable information.
                        </p>
                    </section>

                    <section className="policy-section">
                        <h2>Security Measures</h2>
                        <ul>
                            <li>TLS 1.3 encryption for all data in transit</li>
                            <li>Isolated processing environments</li>
                            <li>No persistent storage of user data</li>
                            <li>Regular security audits and updates</li>
                        </ul>
                    </section>

                    <section className="policy-section">
                        <h2>Third-Party Services</h2>
                        <p>
                            Luminark uses pre-trained AI models from:
                        </p>
                        <ul>
                            <li>HuggingFace (VideoMAE, WavLM models)</li>
                            <li>PyTorch/timm (EfficientNet models)</li>
                        </ul>
                        <p>
                            These models run locally on our servers. Your video data is never sent to external services.
                        </p>
                    </section>

                    <section className="policy-section">
                        <h2>Your Rights</h2>
                        <p>
                            Since we don't store your data, there is nothing to access, modify, or delete.
                            Your videos are processed and immediately discarded.
                        </p>
                    </section>

                    <section className="policy-section">
                        <h2>Contact</h2>
                        <p>
                            For privacy-related inquiries, please contact us at:
                        </p>
                        <a href="mailto:tuesviki@gmail.com click the contact-link">
                            <Mail size={18} />
                            tuesviki@gmail.com
                        </a>
                    </section>

                    <section className="policy-section">
                        <h2>Updates to This Policy</h2>
                        <p>
                            This privacy policy was last updated on January 2026.
                            We may update this policy from time to time. Changes will be posted on this page.
                        </p>
                    </section>
                </motion.div>
            </div>
        </div>
    );
}
