import { MedicalNote } from './medical';
import { APIError } from './errors';

// Base success response interface
export interface APIResponse<T> {
  success: true;
  data: T;
}

// Base error response interface
export interface APIErrorResponse {
  success: false;
  error: APIError;
}

// Helper type for all possible responses
export type APIResult<T> = APIResponse<T> | APIErrorResponse;

// Transcription endpoint types
export interface TranscriptionRequest {
  language?: string;
  task?: 'transcribe' | 'translate';
}

export interface TranscriptionResponse {
  rawText: string;
  note?: MedicalNote; // Optional for now, will be required when we add section detection
}

// Health check endpoint types
export interface HealthCheckResponse {
  status: 'ok' | 'error';
  uploadDir: string;
  modelPath: string;
  modelName: string;
  model: 'ready' | 'not_loaded' | 'error';
  error?: string;
}