import React, { useRef, useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import type { EncodedFrameData } from '../utils/frameEncoder';

interface ARCameraViewProps {
  onFrameCaptured?: (frameData: EncodedFrameData) => void;
  isActive?: boolean;
}

export const ARCameraView: React.FC<ARCameraViewProps> = ({
  onFrameCaptured,
  isActive = true,
}) => {
  const device = useCameraDevice('back');
  const camera = useRef<any>(null);
  const isCapturing = useRef(false);

  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(async () => {
      if (isCapturing.current || !camera.current) return;
      isCapturing.current = true;

      try {
        const image = await camera.current.takeSnapshot();
        const encoded = image.toEncodedImageData('jpg', 70);

        onFrameCaptured?.({
          buffer: encoded.buffer,
          width: encoded.width,
          height: encoded.height,
        });
      } catch (err) {
        console.error('[Camera] Snapshot error:', err);
      } finally {
        isCapturing.current = false;
      }
    }, 200);

    return () => clearInterval(interval);
  }, [isActive, onFrameCaptured]);

  if (!device) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorText}>No camera device available</Text>
      </View>
    );
  }

  return (
    <Camera
      ref={camera}
      style={StyleSheet.absoluteFill}
      device={device}
      isActive={isActive}
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
