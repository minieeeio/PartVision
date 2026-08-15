import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import CameraScreen from './src/screens/CameraScreen';

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <CameraScreen />
    </SafeAreaProvider>
  );
}
