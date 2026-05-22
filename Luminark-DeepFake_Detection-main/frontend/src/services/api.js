/**
 * Luminark API Service
 * 
 * Handles communication with the backend API.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || 'lum_test_key_12345';

/**
 * Analyze a video file for deepfake detection.
 * 
 * @param {File} videoFile - The video file to analyze
 * @param {Object} options - Optional configuration
 * @param {boolean} options.detailed - Get detailed results
 * @returns {Promise<Object>} Analysis result
 */
export async function analyzeVideo(videoFile, options = {}) {
    const endpoint = options.detailed ? '/explain' : '/infer';

    const formData = new FormData();
    formData.append('video', videoFile);

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
            'X-API-Key': API_KEY,
        },
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(
            error.detail?.error ||
            error.detail?.message ||
            `Analysis failed (${response.status})`
        );
    }

    return response.json();
}

/**
 * Check API health status.
 * 
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
        throw new Error('API is not healthy');
    }

    return response.json();
}

/**
 * Get API info.
 * 
 * @returns {Promise<Object>} API information
 */
export async function getApiInfo() {
    const response = await fetch(`${API_BASE_URL}/`);
    return response.json();
}
