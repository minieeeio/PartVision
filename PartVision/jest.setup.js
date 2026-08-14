// Mock external native modules that aren't needed in tests
jest.mock('expo', () => ({}));
jest.mock('expo-constants', () => ({
  __esModule: true,
  default: {
    manifest: { extra: {} },
    expoConfig: { extra: {} },
    manifest2: { extra: {} },
  },
}));
jest.mock('expo-modules-core', () => ({}));
