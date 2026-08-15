import React, { useEffect, useState } from 'react';
import { View, StyleSheet, Text, ActivityIndicator } from 'react-native';
import { Camera } from 'react-native-vision-camera';
import { useObjectDetection } from '../hooks/useObjectDetection';
import BoundingBoxOverlay from '../components/BoundingBoxOverlay';

type PermissionStatus = 'not-determined' | 'denied' | 'authorized';

export default function CameraScreen() {
  const [permission, setPermission] = useState<PermissionStatus>('not-determined');
  const [frameDims, setFrameDims] = useState({ width: 0, height: 0 });

  const {
    detections,
    processTime,
    wsStatus,
    apiBaseUrl,
    configError,
    frameProcessor,
    device,
  } = useObjectDetection();

  useEffect(() => {
    Camera.requestCameraPermission().then(setPermission);
  }, []);

  useEffect(() => {
    if (device?.formats && device.formats.length > 0) {
      const format = device.formats[0];
      setFrameDims({ width: format.videoWidth, height: format.videoHeight });
    }
  }, [device]);

  const requestPermission = async () => {
    const status = await Camera.requestCameraPermission();
    setPermission(status);
  };

  if (permission === 'denied') {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>Camera permission denied</Text>
        <Text style={styles.subText}>Please enable camera access in settings.</Text>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#0066cc" />
        <Text style={styles.loadingText}>Loading camera...</Text>
      </View>
    );
  }

  if (configError) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>Configuration Error</Text>
        <Text style={styles.subText}>{configError}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        frameProcessorFps={5}
        resizeMode="cover"
      />
      <BoundingBoxOverlay
        detections={detections}
        frameWidth={frameDims.width}
        frameHeight={frameDims.height}
      />
      <View style={styles.hud}>
        <View style={styles.hudItem}>
          <Text style={styles.hudLabel}>WS</Text>
          <Text
            style={[
              styles.hudValue,
              { color: wsStatus === 'connected' ? '#00ff88' : '#ff4444' },
            ]}
          >
            {wsStatus}
          </Text>
        </View>
        {processTime > 0 && (
          <View style={styles.hudItem}>
            <Text style={styles.hudLabel}>Latency</Text>
            <Text style={styles.hudValue}>{processTime.toFixed(0)}ms</Text>
          </View>
        )}
        {apiBaseUrl ? (
          <View style={styles.hudItem}>
            <Text style={styles.hudLabel}>Backend</Text>
            <Text style={styles.hudValue} numberOfLines={1}>
              {apiBaseUrl}
            </Text>
          </View>
        ) : null}
      </View>
      {permission === 'not-determined' && (
        <View style={styles.permissionOverlay}>
          <Text style={styles.permissionText}>Camera access required</Text>
          <Text style={styles.permissionSubText} onPress={requestPermission}>
            Tap to grant permission
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
    padding: 24,
  },
  errorText: {
    color: '#ff4444',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
    textAlign: 'center',
  },
  subText: {
    color: '#888',
    fontSize: 14,
    textAlign: 'center',
  },
  loadingText: {
    color: '#fff',
    marginTop: 12,
    fontSize: 16,
  },
  hud: {
    position: 'absolute',
    top: 48,
    left: 16,
    right: 16,
    flexDirection: 'row',
    gap: 16,
    flexWrap: 'wrap',
  },
  hudItem: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  hudLabel: {
    color: '#888',
    fontSize: 10,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  hudValue: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
    marginTop: 2,
  },
  permissionOverlay: {
    position: 'absolute',
    bottom: 80,
    left: 24,
    right: 24,
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
  },
  permissionText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  permissionSubText: {
    color: '#0066cc',
    fontSize: 14,
    fontWeight: '500',
  },
});
