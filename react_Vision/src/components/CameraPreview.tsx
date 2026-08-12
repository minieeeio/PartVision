import React from 'react';
import { StyleSheet } from 'react-native';
import { Camera, Frame } from 'react-native-vision-camera';
import type { CameraDevice } from 'react-native-vision-camera';

export interface CameraPreviewProps {
  device: CameraDevice;
  frameProcessor?: (frame: Frame) => void;
}

export default function CameraPreview({ device, frameProcessor }: CameraPreviewProps) {
  return (
    <Camera
      style={styles.camera}
      device={device}
      isActive={true}
      frameProcessor={frameProcessor}
      frameProcessorFps={15}
    />
  );
}

const styles = StyleSheet.create({
  camera: {
    ...StyleSheet.absoluteFillObject,
  },
});
