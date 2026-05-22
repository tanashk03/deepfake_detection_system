import { Github, Mail } from 'lucide-react';
import './Footer.css';

export default function Footer() {
    return (
        <footer className="footer">
            <div className="footer-container">
                <div className="footer-content">
                    <p className="copyright">
                        © 2026 Luminark. Developed by <strong>Vikas Sharma</strong>
                    </p>
                    <div className="footer-links">
                        <a
                            href="mailto:tuesviki@gmail.com"
                            className="footer-link"
                            title="Email"
                        >
                            <Mail size={16} />
                            tuesviki@gmail.com
                        </a>
                        <a
                            href="https://github.com/IsVohi/Luminark-DeepFake_Detection"
                            className="footer-link"
                            target="_blank"
                            rel="noopener noreferrer"
                            title="GitHub Repository"
                        >
                            <Github size={16} />
                            GitHub
                        </a>
                    </div>
                </div>
            </div>
        </footer>
    );
}
