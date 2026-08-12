import React, { useEffect, useRef, useState, useCallback } from 'react';
import { StyleSheet, View, Alert } from 'react-native';
import {
  Camera,
  useCameraDevices,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { runOnJS } from 'react-native-worklets';

import WebSocketManager from '../managers/WebSocketManager';
import { CameraManager } from '../managers/CameraManager';
import { configManager } from '../managers/ConfigManager';
import {
  PartDetection,
  ConnectionState,
} from '../models/DetectionModel';
import CameraPreview from '../components/CameraPreview';
import BoundingBoxOverlay from '../components/BoundingBoxOverlay';
import ScannerHUD from '../components/ScannerHUD';

const cameraManager = new CameraManager(640, 0.5);
const wsManager = new WebSocketManager('ws://localhost:8000/ws/segment');

export default function ScannerScreen() {
  const [hasPermission, setHasPermission] = useState(false);
  const [cameraPosition, setCameraPosition] = useState<'back' | 'front'>('back');
  const [detections, setDetections] = useState<PartDetection[]>([]);
  const [wsState, setWsState] = useState<ConnectionState>('disconnected');
  const [statusText, setStatusText] = useState<string>('Loading config...');
  const devices = useCameraDevices();
  const lastSendTime = useRef(0);
  const frameCounter = useRef(0);
  const configLoaded = useRef(false);

  // --- Camera permission ---
  useEffect(() => {
    checkPermission();
  }, []);

  const checkPermission = useCallback(async () => {
    const status = await Camera.requestCameraPermission();
    setHasPermission(status === 'authorized');
  }, []);

  // --- Load dynamic backend URL from GitHub config, then connect ---
  useEffect(() => {
    if (!hasPermission) return;

    (async () => {
      try {
        const url = await configManager.getBackendUrl();
        wsManager.setUrl(url);
        setStatusText(`Connecting to ${url.replace('ws://', '').replace('wss://', '')}`);
      } catch {
        setStatusText('Using localhost fallback');
      }

      wsManager.connect();
      configLoaded.current = true;

      wsManager.onState((state: ConnectionState) => {
        setWsState(state);
        if (state === 'connected') setStatusText('Connected');
        else if (state === 'connecting') setStatusText('Connecting...');
        else setStatusText('Reconnecting...');
      });

      wsManager.onDetections((dets: PartDetection[]) => {
        setDetections(dets);
      });

      wsManager.onError((msg: string) => {
        setStatusText(msg);
      });
    })();

    return () => {
      wsManager.disconnect();
    };
  }, [hasPermission]);

  // --- Frame pipeline: camera → JPEG → WebSocket ---
  const sendFrameToJS = useCallback((bytes: Uint8Array) => {
    if (!bytes || bytes.length === 0 || wsState !== 'connected') return;

    const now = Date.now();
    frameCounter.current++;

    // Debounce: ~15 FPS max
    if (now - lastSendTime.current < 66) return;
    lastSendTime.current = now;

    wsManager.sendFrame(bytes);
  }, [wsState]);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    if (wsState !== 'connected' || !frame) return;

    const bytes = cameraManager.encoder.encodeJpegFromFrame(frame);
    if (bytes) {
      runOnJS(sendFrameToJS)(bytes);
    }
  }, [wsState, sendFrameToJS]);

  const activeDevice = devices[cameraPosition];
  const isConnected = wsState === 'connected';

  const handleScanTapped = () => {
    if (isConnected) {
      wsManager.disconnect();
      wsManager.connect();
    }
  };

  if (!hasPermission || !activeDevice) {
    return (
      <View style={styles.center}>
        <ScannerHUD
          isConnected={isConnected}
          onScanTapped={handleScanTapped}
          statusText="Requesting camera permission..."
        />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraPreview device={activeDevice} frameProcessor={frameProcessor} />
      <BoundingBoxOverlay detections={detections} />
      <ScannerHUD
        isConnected={isConnected}
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
