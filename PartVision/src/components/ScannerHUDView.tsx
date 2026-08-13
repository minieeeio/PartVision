import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

interface Props {
  isConnected: boolean;
  onScanTapped: () => void;
}

export const ScannerHUDView: React.FC<Props> = ({ isConnected, onScanTapped }) => {
  return (
    <View style={styles.container} pointerEvents="box-none">
      {/* Top Header Badge */}
      <View style={styles.header}>
        <Text style={styles.title}>CORE_SCAN_V1.0</Text>
        <View style={[styles.statusDot, { backgroundColor: isConnected ? '#00FF00' : '#FF0000' }]} />
      </View>

      {/* Bottom Shutter Button */}
      <View style={styles.bottomBar}>
        <TouchableOpacity style={styles.scanButton} onPress={onScanTapped}>
          <Text style={styles.buttonText}>SCANNER</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    justifyContent: 'space-between',
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.75)',
    padding: 12,
    borderRadius: 4,
    marginTop: 20,
  },
  title: {
    color: '#FFFFFF',
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  bottomBar: {
    alignItems: 'center',
    marginBottom: 20,
  },
  scanButton: {
    backgroundColor: '#FFFF00',
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 4,
  },
  buttonText: {
    color: '#000000',
    fontWeight: 'bold',
    fontFamily: 'monospace',
  },
});