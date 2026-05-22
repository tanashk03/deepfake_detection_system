import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { useEffect } from 'react';
import { ThemeProvider } from './styles/ThemeContext';
import Header from './components/Header';
import Footer from './components/Footer';
import AnimatedBackground from './components/AnimatedBackground';
import Landing from './pages/Landing';
import Analyze from './pages/Analyze';
import Docs from './pages/Docs';
import Privacy from './pages/Privacy';
import './styles/global.css';

function ScrollToTop() {
    const { pathname } = useLocation();

    useEffect(() => {
        window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
    }, [pathname]);

    return null;
}

export default function App() {
    return (
        <ThemeProvider>
            <Router>
                <div className="app">
                    <AnimatedBackground />
                    <Header />
                    <ScrollToTop />
                    <main>
                        <Routes>
                            <Route path="/" element={<Landing />} />
                            <Route path="/analyze" element={<Analyze />} />
                            <Route path="/docs" element={<Docs />} />
                            <Route path="/privacy" element={<Privacy />} />
                        </Routes>
                    </main>
                    <Footer />
                </div>
            </Router>
        </ThemeProvider>
    );
}
