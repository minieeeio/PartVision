module.exports = {
  preset: '@react-native/jest-preset',
  testEnvironment: 'node',
  rootDir: '.',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.test.{ts,tsx}'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  transform: {
    '\\.[jt]sx?$': 'babel-jest',
  },
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native|expo|expo-constants|expo-modules-core|react-native-worklets-core|react-native-worklets|react-native-nitro-image|react-native-nitro-modules|react-native-reanimated)/)',
  ],
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleNameMapper: {
    '\\.(png|jpg|jpeg|gif|webp|svg|bmp|ttf|otf|woff|woff2)$': '<rootDir>/src/__mocks__/fileMock.js',
  },
};
