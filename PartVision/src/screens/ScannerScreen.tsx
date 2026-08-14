import React, { useState } from 'react';
import { View, StyleSheet, LayoutRectangle } from 'react-native';
import { useWebSocket } from '../hooks/useWebSocket';
import { ARCameraView } from '../components/ARCameraView';
import { PolygonOverlay } from '../components/PolygonOverlay';
import { ScannerHUDView } from '../components/ScannerHUDView';
import type { LocationData } from '../types/detection';

interface ScannerScreenProps {
  backendUrl: string;
  location: LocationData | null;
}

export const ScannerScreen: React.FC<ScannerScreenProps> = ({ backendUrl, location }) => {
  const [containerSize, setContainerSize] = useState<LayoutRectangle>({ x: 0, y: 0, width: 0, height: 0 });
  const { isConnected, detections, sendFrame, sendLocation } = useWebSocket(backendUrl);

  return (
    <View
      style={styles.container}
      onLayout={(e) => setContainerSize(e.nativeEvent.layout)}
    >
      <ARCameraView onFrameCaptured={sendFrame} />

      <PolygonOverlay
        detections={detections}
        containerSize={containerSize}
      />

      <ScannerHUDView
        isConnected={isConnected}
        location={location}
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
