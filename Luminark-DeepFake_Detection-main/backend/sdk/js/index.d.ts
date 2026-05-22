declare module '@luminark/sdk' {
    export class LuminarkError extends Error {
        code: string;
        statusCode: number;
        constructor(message: string, code: string, statusCode?: number);
    }

    export class LuminarkResult {
        verdict: 'REAL' | 'FAKE' | 'INCONCLUSIVE';
        confidence: number;
        explanation: string;
        processingTimeMs?: number;
        score?: number;
        uncertainty?: number;
        modalityContributions?: Record<string, number>;

        get isFake(): boolean;
        get isReal(): boolean;
        get isInconclusive(): boolean;
    }

    export interface LuminarkOptions {
        baseUrl?: string;
        timeout?: number;
    }

    export interface AnalyzeOptions {
        detailed?: boolean;
    }

    export class Luminark {
        constructor(apiKey?: string, options?: LuminarkOptions);
        analyze(video: string | Buffer, options?: AnalyzeOptions): Promise<LuminarkResult>;
        health(): Promise<{ status: string; version: string }>;
    }

    export function analyze(
        video: string,
        apiKey?: string,
        options?: LuminarkOptions
    ): Promise<LuminarkResult>;
}
