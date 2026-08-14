import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import type { LocationData } from '../types/detection';

interface Props {
  isConnected: boolean;
  location: LocationData | null;
}

export const ScannerHUDView: React.FC<Props> = ({ isConnected, location }) => {
  const lat = location?.latitude != null ? location.latitude.toFixed(6) : '--';
  const lng = location?.longitude != null ? location.longitude.toFixed(6) : '--';
  const acc = location?.accuracy != null ? `±${location.accuracy.toFixed(1)}m` : '--';

  return (
    <View style={styles.container} pointerEvents="box-none">
      <View style={styles.header}>
        <Text style={styles.title}>CORE_SCAN_V1.0</Text>
        <View style={[styles.statusDot, { backgroundColor: isConnected ? '#00FF00' : '#FF0000' }]} />
      </View>
      <View style={styles.locationContainer}>
        <Text style={styles.locationText}>LAT: {lat}</Text>
        <Text style={styles.locationText}>LNG: {lng}</Text>
        <Text style={styles.locationText}>ACC: {acc}</Text>
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
  locationContainer: {
    marginTop: 10,
    backgroundColor: 'rgba(0,0,0,0.75)',
    padding: 10,
    borderRadius: 4,
  },
  locationText: {
    color: '#00FFFF',
    fontFamily: 'monospace',
    fontSize: 12,
  },
});