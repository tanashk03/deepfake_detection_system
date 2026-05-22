import { motion } from 'framer-motion';
import { useTheme } from '../styles/ThemeContext';
import './AnimatedBackground.css';

export default function AnimatedBackground() {
    const { theme } = useTheme();

    return (
        <div className={`animated-background ${theme}`}>
            <div className="gradient-blob blob-1" />
            <div className="gradient-blob blob-2" />
            <div className="gradient-blob blob-3" />
            <div className="bg-grid" />
            <div className="bg-noise" />
        </div>
    );
}
