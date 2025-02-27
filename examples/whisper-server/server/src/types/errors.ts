// Error types for API responses
export type ErrorType = 
  | 'VALIDATION_ERROR'    // Request validation errors
  | 'TRANSCRIPTION_ERROR' // Whisper-specific errors
  | 'FILE_SYSTEM_ERROR'   // File operations errors
  | 'SERVER_ERROR';       // Generic server errors

// Base error structure
export interface APIError {
  type: ErrorType;
  message: string;
  details?: unknown;
  code?: number;
}

// Validation error details
export interface ValidationErrorDetails {
  field: string;
  reason: string;
}

// Transcription error details
export interface TranscriptionErrorDetails {
  audioFile?: string;
  modelName?: string;
  errorCode?: string;
}

// File system error details
export interface FileSystemErrorDetails {
  path?: string;
  operation: 'read' | 'write' | 'delete' | 'access';
  systemError?: string;
}

// Type guard to check if something is an APIError
export const isAPIError = (error: unknown): error is APIError => {
  return (
    typeof error === 'object' &&
    error !== null &&
    'type' in error &&
    'message' in error
  );
};