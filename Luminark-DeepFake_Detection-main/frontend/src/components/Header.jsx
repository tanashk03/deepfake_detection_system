import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, Github, ExternalLink, Shield, Moon, Sun } from 'lucide-react';
import { useTheme } from '../styles/ThemeContext';
import './Header.css';

export default function Header() {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const location = useLocation();
    const { theme, toggleTheme } = useTheme();

    const navLinks = [
        { path: '/', label: 'Home' },
        { path: '/analyze', label: 'Analyze' },
        { path: '/docs', label: 'Docs' },
        { path: '/privacy', label: 'Privacy' },
    ];

    const isActive = (path) => location.pathname === path;

    return (
        <header className="header">
            <div className="header-container">
                {/* Logo */}
                <Link to="/" className="logo">
                    <div className="logo-icon-wrapper">
                        <img
                            src="/assets/logo.png"
                            alt="Luminark Logo"
                            className="logo-icon"
                        />
                    </div>
                    <span className="logo-text">Luminark</span>
                </Link>

                {/* Desktop Navigation */}
                <nav className="nav-desktop">
                    {navLinks.map((link) => (
                        <Link
                            key={link.path}
                            to={link.path}
                            className={`nav-link ${isActive(link.path) ? 'active' : ''}`}
                        >
                            {link.label}
                            {isActive(link.path) && (
                                <motion.div
                                    className="nav-indicator"
                                    layoutId="nav-indicator"
                                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                                />
                            )}
                        </Link>
                    ))}
                </nav>

                {/* Actions */}
                <div className="header-actions">
                    {/* Theme Toggle */}
                    <button
                        className="theme-toggle"
                        onClick={toggleTheme}
                        aria-label="Toggle theme"
                    >
                        <AnimatePresence mode="wait" initial={false}>
                            <motion.div
                                key={theme}
                                initial={{ y: -20, opacity: 0 }}
                                animate={{ y: 0, opacity: 1 }}
                                exit={{ y: 20, opacity: 0 }}
                                transition={{ duration: 0.2 }}
                            >
                                {theme === 'dark' ? <Moon size={20} /> : <Sun size={20} />}
                            </motion.div>
                        </AnimatePresence>
                    </button>

                    <div className="divider-vertical" />

                    {/* External Links */}
                    <div className="external-links">
                        <a
                            href="#"
                            className="external-link"
                            target="_blank"
                            rel="noopener noreferrer"
                            title="Portfolio"
                        >
                            <ExternalLink size={18} />
                            <span>Portfolio</span>
                        </a>
                        <a
                            href="https://github.com/IsVohi/Luminark-DeepFake_Detection"
                            className="external-link github"
                            target="_blank"
                            rel="noopener noreferrer"
                            title="GitHub Repository"
                        >
                            <Github size={18} />
                            <span>GitHub</span>
                        </a>
                    </div>

                    {/* Mobile Menu Button */}
                    <button
                        className="menu-toggle"
                        onClick={() => setIsMenuOpen(!isMenuOpen)}
                        aria-label="Toggle menu"
                    >
                        {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
                    </button>
                </div>
            </div>

            {/* Mobile Navigation */}
            {isMenuOpen && (
                <motion.nav
                    className="nav-mobile"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                >
                    {navLinks.map((link) => (
                        <Link
                            key={link.path}
                            to={link.path}
                            className={`nav-link-mobile ${isActive(link.path) ? 'active' : ''}`}
                            onClick={() => setIsMenuOpen(false)}
                        >
                            {link.label}
                        </Link>
                    ))}
                    <div className="mobile-external">
                        <a href="#" className="external-link" target="_blank" rel="noopener noreferrer">
                            <ExternalLink size={18} />
                            Portfolio
                        </a>
                        <a href="https://github.com/IsVohi/Luminark-DeepFake_Detection" className="external-link" target="_blank" rel="noopener noreferrer">
                            <Github size={18} />
                            GitHub
                        </a>
                    </div>
                </motion.nav>
            )}
        </header>
    );
}
