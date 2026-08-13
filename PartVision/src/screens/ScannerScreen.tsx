import React, { useState } from 'react';
import { View, StyleSheet, LayoutRectangle } from 'react-native';
import { useWebSocket } from '../hooks/useWebSocket';
import { ARCameraView } from '../components/ARCameraView';
import { BoundingBoxView } from '../components/BoundingBoxView';
import { ScannerHUDView } from '../components/ScannerHUDView';

const BACKEND_URL = 'wss://your-fastapi-server.com/ws/segment';

export const ScannerScreen: React.FC = () => {
  const [containerSize, setContainerSize] = useState<LayoutRectangle>({ x: 0, y: 0, width: 0, height: 0 });
  const { isConnected, detections, sendFrame } = useWebSocket(BACKEND_URL);

  return (
    <View
      style={styles.container}
      onLayout={(e) => setContainerSize(e.nativeEvent.layout)}
    >
      {/* 1. Camera Background Stream */}
      <ARCameraView onFrameCaptured={sendFrame} />

      {/* 2. Real-time AI Overlay Layer */}
      {detections.map((detection, index) => (
        <BoundingBoxView
          key={`${detection.label}-${index}`}
          detection={detection}
          containerSize={containerSize}
        />
      ))}

      {/* 3. Top-level HUD Interface */}
      <ScannerHUDView
        isConnected={isConnected}
        onScanTapped={() => console.log('Scan action triggered')}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
});