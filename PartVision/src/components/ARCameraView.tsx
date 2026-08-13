import React from 'react';
import { StyleSheet, View, Text } from 'react-native';
import { 
  Camera, 
  useCameraDevice, 
  useFrameProcessor, 
  Frame 
} from 'react-native-vision-camera';
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

  const frameProcessor = useFrameProcessor((frame: Frame) => {
    'worklet';
    const encodedData = encodeFrameToBase64(frame);
    if (encodedData && onFrameCaptured) {
      // Execute frame streaming callback
    }
  }, [onFrameCaptured]);

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
      frameProcessor={frameProcessor}
      pixelFormat="yuv"
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