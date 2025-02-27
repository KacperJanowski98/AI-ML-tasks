// Add custom matchers for TypeScript type checking
expect.extend({
  toBeType(received: unknown, type: string) {
    const pass = typeof received === type;
    if (pass) {
      return {
        message: () => `expected ${received} not to be type ${type}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be type ${type}`,
        pass: false,
      };
    }
  },
});

// Extend Jest matchers
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeType(type: string): R;
    }
  }
}