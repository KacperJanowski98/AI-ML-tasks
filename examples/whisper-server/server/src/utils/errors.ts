import { APIError, ErrorType, ValidationErrorDetails, TranscriptionErrorDetails, FileSystemErrorDetails } from '../types/errors';

// Error factory for creating typed API errors
export const createError = (
  type: ErrorType,
  message: string,
  details?: unknown,
  code?: number
): APIError => ({
  type,
  message,
  details,
  code,
});

// Validation error helper
export const createValidationError = (
  field: string,
  reason: string
): APIError => {
  const details: ValidationErrorDetails = { field, reason };
  return createError('VALIDATION_ERROR', `Validation failed for ${field}`, details, 400);
};

// Transcription error helper
export const createTranscriptionError = (
  message: string,
  details: Partial<TranscriptionErrorDetails>
): APIError => {
  return createError('TRANSCRIPTION_ERROR', message, details, 500);
};

// File system error helper
export const createFileSystemError = (
  operation: FileSystemErrorDetails['operation'],
  path?: string,
  systemError?: string
): APIError => {
  const details: FileSystemErrorDetails = {
    operation,
    path,
    systemError
  };
  return createError('FILE_SYSTEM_ERROR', `File system ${operation} operation failed`, details, 500);
};