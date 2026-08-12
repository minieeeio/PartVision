import React, { useEffect, useState } from 'react';
import { StyleSheet, View, Text, TouchableOpacity } from 'react-native';

export interface ScannerHUDProps {
  isConnected: boolean;
  onScanTapped: () => void;
  statusText?: string;
}

export default function ScannerHUD({
  isConnected,
  onScanTapped,
  statusText,
}: ScannerHUDProps) {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>CORE_SCAN_V1.0</Text>
        <View style={styles.statusRow}>
          <View
            style={[
              styles.statusDot,
              { backgroundColor: isConnected ? '#00ff00' : '#ff0000' },
            ]}
          />
          {statusText ? (
            <Text style={styles.statusLabel}>{statusText}</Text>
          ) : null}
        </View>
      </View>

      <View style={styles.bottomBar}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={onScanTapped}
          activeOpacity={0.8}
        >
          <View style={styles.actionButtonInner}>
            <View style={styles.actionIcon} />
            <Text style={styles.actionLabel}>SCANNER</Text>
          </View>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
    pointerEvents: 'box-none',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    marginHorizontal: 16,
    marginTop: 10,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  headerText: {
    fontSize: 18,
    fontWeight: '900',
    fontFamily: 'monospace',
    letterSpacing: 1.5,
    color: '#000',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  statusLabel: {
    fontSize: 11,
    fontWeight: '600',
    fontFamily: 'monospace',
    color: '#000',
  },
  bottomBar: {
    alignItems: 'center',
    paddingBottom: 20,
  },
  actionButton: {
    width: 110,
    height: 54,
    backgroundColor: '#ffcc00',
    borderWidth: 2.5,
    borderColor: '#000',
    borderRadius: 0,
    shadowColor: '#000',
    shadowOffset: { width: 3, height: 3 },
    shadowOpacity: 0.4,
    shadowRadius: 0,
    elevation: 4,
  },
  actionButtonInner: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  actionIcon: {
    width: 18,
    height: 18,
    backgroundColor: '#000',
    borderRadius: 2,
  },
  actionLabel: {
    fontSize: 11,
    fontWeight: '900',
    fontFamily: 'monospace',
    color: '#000',
  },
});
