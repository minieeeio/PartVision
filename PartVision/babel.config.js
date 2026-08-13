module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      'react-native-worklets-core/plugin', // Required for VisionCamera Frame Processors
      'react-native-reanimated/plugin',    // Reanimated plugin (must always be last)
    ],
  };
};