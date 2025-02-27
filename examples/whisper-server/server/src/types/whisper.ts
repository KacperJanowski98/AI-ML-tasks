export interface TranscriptionResult {
  text: string;
  segments: Array<{
    text: string;
    start: number;
    end: number;
    confidence: number;
  }>;
  language: string;
}

export interface TranscriptionOptions {
  modelName: string;  // Added to match actual usage
  language?: string;
  task?: 'transcribe' | 'translate';
  timestamps?: boolean;
  word_timestamps?: boolean;
}