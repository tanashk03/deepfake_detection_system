/**
 * Luminark JavaScript SDK
 * 
 * One-line deepfake detection.
 * 
 * @example
 * import { Luminark } from '@luminark/sdk';
 * 
 * const client = new Luminark('your_api_key');
 * const result = await client.analyze('./video.mp4');
 * 
 * console.log(result.verdict);      // "FAKE"
 * console.log(result.confidence);   // 87
 * console.log(result.explanation);  // "..."
 */

const fs = require('fs');
const path = require('path');
const FormData = require('form-data');

/**
 * Luminark API error
 */
class LuminarkError extends Error {
    constructor(message, code, statusCode = 0) {
        super(message);
        this.name = 'LuminarkError';
        this.code = code;
        this.statusCode = statusCode;
    }
}

/**
 * Result from deepfake analysis
 */
class LuminarkResult {
    constructor(data) {
        this.verdict = data.verdict;
        this.confidence = data.confidence;
        this.explanation = data.explanation;
        this.processingTimeMs = data.processing_time_ms;

        // Extended fields (from /explain)
        this.score = data.score;
        this.uncertainty = data.uncertainty;
        this.modalityContributions = data.modality_contributions;
    }

    get isFake() {
        return this.verdict === 'FAKE';
    }

    get isReal() {
        return this.verdict === 'REAL';
    }

    get isInconclusive() {
        return this.verdict === 'INCONCLUSIVE';
    }
}

/**
 * Luminark API client
 */
class Luminark {
    static DEFAULT_BASE_URL = 'http://localhost:8000';

    /**
     * Create a Luminark client
     * @param {string} apiKey - API key
     * @param {Object} options - Configuration options
     * @param {string} [options.baseUrl] - API base URL
     * @param {number} [options.timeout] - Request timeout in ms
     */
    constructor(apiKey, options = {}) {
        if (!apiKey) {
            apiKey = process.env.LUMINARK_API_KEY;
        }

        if (!apiKey) {
            throw new LuminarkError(
                'API key required. Pass apiKey or set LUMINARK_API_KEY',
                'MISSING_API_KEY'
            );
        }

        this.apiKey = apiKey;
        this.baseUrl = (options.baseUrl || process.env.LUMINARK_API_URL ||
            Luminark.DEFAULT_BASE_URL).replace(/\/$/, '');
        this.timeout = options.timeout || 300000; // 5 minutes
    }

    /**
     * Analyze a video for deepfake indicators
     * @param {string|Buffer} video - Path to video file or Buffer
     * @param {Object} options - Analysis options
     * @param {boolean} [options.detailed=false] - Include internal scores
     * @returns {Promise<LuminarkResult>}
     */
    async analyze(video, options = {}) {
        const endpoint = options.detailed ? '/explain' : '/infer';

        const form = new FormData();

        if (typeof video === 'string') {
            // File path
            if (!fs.existsSync(video)) {
                throw new LuminarkError(
                    `Video file not found: ${video}`,
                    'FILE_NOT_FOUND'
                );
            }
            form.append('video', fs.createReadStream(video), path.basename(video));
        } else if (Buffer.isBuffer(video)) {
            // Raw bytes
            form.append('video', video, { filename: 'video.mp4' });
        } else {
            throw new LuminarkError('Video must be a file path or Buffer', 'INVALID_INPUT');
        }

        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'X-API-Key': this.apiKey,
                    ...form.getHeaders(),
                },
                body: form,
                signal: AbortSignal.timeout(this.timeout),
            });

            const data = await response.json();

            if (response.status === 401) {
                throw new LuminarkError(
                    data.detail?.error || 'Authentication failed',
                    'AUTH_ERROR',
                    401
                );
            }

            if (!response.ok) {
                throw new LuminarkError(
                    data.detail?.error || 'Request failed',
                    data.detail?.code || 'UNKNOWN_ERROR',
                    response.status
                );
            }

            return new LuminarkResult(data);

        } catch (error) {
            if (error instanceof LuminarkError) throw error;

            if (error.name === 'AbortError') {
                throw new LuminarkError('Request timeout', 'TIMEOUT');
            }

            throw new LuminarkError(
                `Connection error: ${error.message}`,
                'CONNECTION_ERROR'
            );
        }
    }

    /**
     * Check API health
     * @returns {Promise<Object>}
     */
    async health() {
        const response = await fetch(`${this.baseUrl}/health`);
        return response.json();
    }
}

/**
 * One-shot analysis without managing client
 * @param {string} video - Path to video file
 * @param {string} apiKey - API key
 * @param {Object} [options] - Options
 * @returns {Promise<LuminarkResult>}
 */
async function analyze(video, apiKey, options = {}) {
    const client = new Luminark(apiKey, options);
    return client.analyze(video);
}

module.exports = {
    Luminark,
    LuminarkResult,
    LuminarkError,
    analyze,
};
