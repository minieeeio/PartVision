import React, { useEffect, useState, useCallback } from 'react';
import { StyleSheet, View, Alert } from 'react-native';
import {
  Camera,
  useCameraDevices,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { runOnJS, useSharedValue } from 'react-native-reanimated';

import { inferenceManager } from '../managers/InferenceManager';
import { PartDetection } from '../models/DetectionModel';
import CameraPreview from '../components/CameraPreview';
import BoundingBoxOverlay from '../components/BoundingBoxOverlay';
import ScannerHUD from '../components/ScannerHUD';
import { INFERENCE_CONFIG } from '../utils/constants';

export default function ScannerScreen() {
  const [hasPermission, setHasPermission] = useState(false);
  const [cameraPosition, setCameraPosition] = useState<'back' | 'front'>('back');
  const [detections, setDetections] = useState<PartDetection[]>([]);
  const [statusText, setStatusText] = useState<string>('Loading model...');
  const devices = useCameraDevices();
  const lastInferenceTime = useSharedValue(0);
  const modelLoaded = useSharedValue(false);

  useEffect(() => {
    (async () => {
      setStatusText('Loading ONNX model...');
      const success = await inferenceManager.load();
      if (success) {
        modelLoaded.value = true;
        setStatusText('Ready — point camera at a car');
      } else {
        modelLoaded.value = false;
        Alert.alert(
          'Model load failed',
          'Place partlite_unet.onnx in react_Vision/src/assets/',
        );
        setStatusText('Model load failed');
      }
    })();

    return () => {
      inferenceManager.dispose();
    };
  }, []);

  const checkPermission = useCallback(async () => {
    const status = await Camera.requestCameraPermission();
    setHasPermission(status === 'authorized');
  }, []);

  useEffect(() => {
    checkPermission();
  }, [checkPermission]);

  const processFrame = useCallback((rgba: Uint8Array, width: number, height: number) => {
    const now = Date.now();
    if (now - lastInferenceTime.value < 166) return;
    lastInferenceTime.value = now;

    inferenceManager
      .infer(rgba, width, height)
      .then((dets) => setDetections(dets))
      .catch((e) => console.error('[ScannerScreen] Inference error:', e));
  }, []);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    if (!modelLoaded.value || !frame) return;

    const rgba = frame.toRGBA();
    if (rgba) {
      runOnJS(processFrame)(rgba, frame.width, frame.height);
    }
  }, [processFrame]);

  const activeDevice = devices[cameraPosition];

  const handleScanTapped = () => {
    setCameraPosition((prev) => (prev === 'back' ? 'front' : 'back'));
  };

  if (!hasPermission || !activeDevice) {
    return (
      <View style={styles.center}>
        <ScannerHUD
          isConnected={true}
          onScanTapped={handleScanTapped}
          statusText="Requesting camera permission..."
        />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraPreview
        device={activeDevice}
        frameProcessor={frameProcessor}
        frameProcessorFps={INFERENCE_CONFIG.INFERENCE_FPS}
      />
      <BoundingBoxOverlay detections={detections} />
      <ScannerHUD
        isConnected={modelLoaded.value}
        onScanTapped={handleScanTapped}
        statusText={statusText}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
});
