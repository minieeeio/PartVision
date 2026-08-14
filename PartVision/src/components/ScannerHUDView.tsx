import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface Props {
  isConnected: boolean;
}

export const ScannerHUDView: React.FC<Props> = ({ isConnected }) => {
  return (
    <View style={styles.container} pointerEvents="box-none">
      <View style={styles.header}>
        <Text style={styles.title}>CORE_SCAN_V1.0</Text>
        <View style={[styles.statusDot, { backgroundColor: isConnected ? '#00FF00' : '#FF0000' }]} />
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
    justifyContent: 'flex-start',
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
});