import React from 'react';
import { StyleSheet } from 'react-native';
import { Camera, Frame, CameraDevice } from 'react-native-vision-camera';

export interface CameraPreviewProps {
  device: CameraDevice;
  frameProcessor?: (frame: Frame) => void;
  frameProcessorFps?: number;
}

export default function CameraPreview({
  device,
  frameProcessor,
  frameProcessorFps = 6,
}: CameraPreviewProps) {
  return (
    <Camera
      style={styles.camera}
      device={device}
      isActive={true}
      frameProcessor={frameProcessor}
      frameProcessorFps={frameProcessorFps}
    />
  );
}

const styles = StyleSheet.create({
  camera: {
    ...StyleSheet.absoluteFillObject,
  },
});
