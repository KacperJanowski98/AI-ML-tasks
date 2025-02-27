import { join } from 'path';
import { getPath, PATHS, SERVER_ROOT } from '../../utils/paths';

describe('Paths Utilities', () => {
  describe('SERVER_ROOT', () => {
    it('should be a valid path string', () => {
      expect(SERVER_ROOT).toBeType('string');
      expect(SERVER_ROOT).toContain('server');
    });
  });

  describe('PATHS', () => {
    it('should define required directories', () => {
      expect(PATHS).toHaveProperty('uploads');
      expect(PATHS).toHaveProperty('models');
    });

    it('should create paths relative to SERVER_ROOT', () => {
      expect(PATHS.uploads).toBe(join(SERVER_ROOT, 'uploads'));
      expect(PATHS.models).toBe(join(SERVER_ROOT, 'models'));
    });
  });

  describe('getPath', () => {
    it('should return base directory path when no file provided', () => {
      expect(getPath('uploads')).toBe(PATHS.uploads);
      expect(getPath('models')).toBe(PATHS.models);
    });

    it('should return full file path when file provided', () => {
      const testFile = 'test.txt';
      expect(getPath('uploads', testFile)).toBe(join(PATHS.uploads, testFile));
      expect(getPath('models', testFile)).toBe(join(PATHS.models, testFile));
    });

    it('should handle nested paths correctly', () => {
      const nestedPath = 'folder/subfolder/file.txt';
      expect(getPath('uploads', nestedPath)).toBe(join(PATHS.uploads, nestedPath));
    });

    // TypeScript compiler will catch this, but good to test runtime behavior
    it('should handle all defined path types', () => {
      const pathTypes = Object.keys(PATHS) as Array<keyof typeof PATHS>;
      pathTypes.forEach(type => {
        expect(() => getPath(type)).not.toThrow();
      });
    });
  });
});