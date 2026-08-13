import React from 'react';
import { StyleSheet, View, Text } from 'react-native';
import { 
  Camera, 
  useCameraDevice, 
  useFrameOutput, 
  Frame 
} from 'react-native-vision-camera';
import { useRunOnJS } from 'react-native-worklets-core';
import { encodeFrameToBase64 } from '../utils/frameEncoder';

interface ARCameraViewProps {
  onFrameCaptured?: (frameData: string) => void;
  isActive?: boolean;
}

export const ARCameraView: React.FC<ARCameraViewProps> = ({ 
  onFrameCaptured, 
  isActive = true 
}) => {
  const device = useCameraDevice('back');

  // 1. Memoize the JS callback so it can safely be called inside C++ Worklets
  const handleFrameCapturedOnJS = useRunOnJS((encodedData: string) => {
    if (onFrameCaptured) {
      onFrameCaptured(encodedData);
    }
  }, [onFrameCaptured]);

  // 2. High-performance frame listener
  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    onFrame: (frame: Frame) => {
      'worklet';
      try {
        const encodedData = encodeFrameToBase64(frame);
        
        if (encodedData) {
          // Call the memoized thread wrapper without re-instantiating functions
          handleFrameCapturedOnJS(encodedData);
        }
      } catch (error) {
        // Handle worklet frame processing error
      } finally {
        // ALWAYS release native frame buffer to avoid memory leaks
        frame.dispose();
      }
    },
  });

  if (!device) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorText}>No camera device available</Text>
      </View>
    );
  }

  return (
    <Camera
      style={StyleSheet.absoluteFill}
      device={device}
      isActive={isActive}
      outputs={[frameOutput]}
    />
  );
};

const styles = StyleSheet.create({
  errorContainer: {
    flex: 1,
    backgroundColor: '#000000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorText: {
    color: '#FF3366',
    fontSize: 14,
    fontFamily: 'monospace',
  },
});